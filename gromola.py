"""
This is the central Module which is a class from which the different functions are called
WEAKNESSES:
    CURRENTLY WE HAVE TO WRITE OUT THE manipulated DEM and read it in TO START THE MESHING
    -pyvista export has problems if cell data array and point data have the same length, it automatically associates the cell data array as point data :-()

"""
import os
import yaml
import numpy as np
import pyvista as pv
import secrets
import geopandas as gpd
import pandas as pd

from shutil import copyfile
from copy import deepcopy
from datetime import datetime,timedelta
from .gisprocessing.stream_simplification import simplify_network as _simplify_network
from .gisprocessing.stream_simplification import simplify_network_eu as _simplify_network_eu


from .gisprocessing.util_functions import time_reindexing as _time_reindexing

from .gisprocessing.basin_delineation import delineate_basins as _delineate_basins
from .gisprocessing.basin_delineation import delineate_basins_eu as _delineate_basins_eu
from .gisprocessing.inner_surface_processing import process_lakes as _process_lakes
from .gisprocessing.inner_surface_processing import process_mines as _process_mines
from .gisprocessing.inner_surface_processing import clip_observation_wells as _clip_observation_wells
from .gisprocessing.burning_mapping import burn_lakes as _burn_lakes
from .gisprocessing.burning_mapping import burn_streams as _burn_streams
from .gisprocessing.burning_mapping import map_streams as _map_streams
from .gisprocessing.burning_mapping import map_streams_eu as _map_streams_eu
from.calibration.pest import pest as _pest



from .meshing.meshing import mesh_processing_2d as _mesh_processing_2d
from .meshing.meshing import mesh_processing_3d as _mesh_processing_3d


from .simulation.opengeosys.opengeosys import ogs_simulator as _ogs_simulator
from .simulation.modflow.modflow import modflow_simulator as _modflow_simulator

import warnings
from pandas.core.common import SettingWithCopyWarning

def iterdict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        elif isinstance(v, list):
            try:
                v=list(map(float,v))
            except:
                pass
            d.update({k: v})
        else:
            if type(v) == str:
                try:
                    v=float(v)
                except:
                    pass
            d.update({k: v})
    return d


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
print('All DeprectiationWarnings are ignored, be careful')

class model:
    def __init__(self,configfile,model_token=None,output_dir=None):
        
        #first we generate a new token and create an output directory
        #generate a 4 byte token to identify each run
        if model_token is None:
            self.identifier=secrets.token_hex(nbytes=4)
        else:
            self.identifier=model_token
            
            
        """
        if self.simulation_token is None:  
            print('Create a new token')
            self.identifier=secrets.token_hex(nbytes=4)
        else:
            print('Take simulation token to identify output dir')
            self.identifier=self.simulation_token
        #add to the output the identifier
        """
        #generate an output dir
        self.model_path=os.getcwd()
        if output_dir is None:
            print('Define ',os.path.join(self.model_path,'output',self.identifier), 'as output directory')
            self.output_dir=os.path.join(self.model_path,'output',self.identifier)
        else:
            self.output_dir=os.path.join(output_dir,self.identifier)
        
        
        os.makedirs(self.output_dir,exist_ok=True)
        
        #if there is no gromola.yml file in the output dir, we copy it there
        if os.path.split(configfile)[-1] not in os.listdir(self.output_dir):
            copyfile(configfile, os.path.join(self.output_dir,os.path.split(configfile)[-1]))
        self.config_path=os.path.join(self.output_dir,os.path.split(configfile)[-1])
        
        #%% read the config file
        with open(self.config_path) as c:
            self.config = yaml.safe_load(c)
        #fix potentially badl
        iterdict(self.config)
        # copy the calibration configuration file 
        calibration_config_file=self.config['calibration']['calibration_config_path']
        
        
        #same for the yml file
        if 'calibration_config.csv' not in os.listdir(self.output_dir):
            copyfile(os.path.join(self.model_path,calibration_config_file), os.path.join(self.output_dir,calibration_config_file))
            
            
        #same the configuration to the run directory
        with open(os.path.join(self.output_dir,'run_config.yml'), 'w') as configfile:
            yaml.dump(self.config, configfile, default_flow_style=False)
            

        #repair the data path
        if not os.path.isabs(self.config['input']['data_path']):
            print(self.config['input']['data_path'], 'is relative path, append gromola directory')
            self.config['input']['data_path']=os.path.join(os.getcwd(),self.config['input']['data_path'])
            
            
        #get the start year
        self.start_year=str(self.config['simulation']['time']['start_date'])[:4]
        
        #get information on the GIS processing, whether it was done already
        self.gis_processed=False
        self.mesh_processed=False
        
        
        
        
    def simplify_streams(self):
        """
        Calls the simplify network functionality and returns a dictionary with the streams

        Returns
        -------
        None.
        """
        stream_input_cfg= self.config['input']['hydrology']['streams']
        stream_geo_cfg = self.config['discretization']['geometry']['streams']
        
        if len(stream_geo_cfg['longitudinal'])>1:
            print('Currently, resolution changes with stream order are not implemented')
            
        l_vertex=stream_geo_cfg['longitudinal'][0]
        
        self.config['discretization']['geometry']['streams']
        
        if stream_input_cfg['network_source']=='Landesamt':
            self.streams_planar=_simplify_network(l_vertex=l_vertex,
                                              maximum_stream_order=stream_input_cfg['threshold_stream_order'],
                                              streams=stream_input_cfg['networks'],
                                              data_path=os.path.join(self.config['input']['data_path'],
                                                                     stream_input_cfg['subpath'],
                                                                     stream_input_cfg['network_source']),
                                              output_path=self.output_dir)
            
        elif stream_input_cfg['network_source']=='Hydro-EU':
            self.streams_planar=_simplify_network_eu(l_vertex=l_vertex,
                                              minimum_strahler_order=stream_input_cfg['threshold_stream_order'],
                                              streams=stream_input_cfg['networks'],
                                              data_path=os.path.join(self.config['input']['data_path'],
                                                                     stream_input_cfg['subpath'],
                                                                     stream_input_cfg['network_source']),
                                              gauge_dir=os.path.join(self.config['input']['data_path'],
                                                                     self.config['input']['hydrology']['gauges']['subpath']),
                                              gauges=self.config['input']['hydrology']['gauges']['gauge_names'],
                                              output_path=self.output_dir
                                              )
            
        else:
            raise KeyError('CurrentlyNetwork Type can be either Landesamt or Hydro-EU')
        

    def delineate_basin(self,export_watershed_analysis=False,sharp_edge_ratio=0.5,
                        min_search_factor=1/3,
                        max_search_factor=5/3,
                        ):
        """
        

        Parameters
        ----------

        sharp_edge_ratio : TYPE, optional
            DESCRIPTION. The default is 0.5.
        export_watershed_analysis : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        #%% Compute the search distances
        resolutions=[self.config['discretization']['geometry'][x] for x in ['domain','mines','lakes']]
        resolutions.append(min(self.config['discretization']['geometry']['streams']['longitudinal']))
        
        min_pntsearch_distance=min(resolutions)*min_search_factor
        max_pntsearch_distance=min(resolutions)*max_search_factor
        
        
        #%% Call the processing unit       
        if self.config['input']['hydrology']['streams']['network_source']=='Landesamt':
            self.domain,self.streams_sorted=_delineate_basins(l_vertex=self.config['discretization']['geometry']['domain'],
                                                              dem_path=os.path.join(self.config['input']['data_path'],
                                                                                    self.config['input']['topography']['dem_path']),
                                                              stream_networks=self.streams_planar,
                                                              gauge_dir=os.path.join(self.config['input']['data_path'],
                                                                                     self.config['input']['hydrology']['gauges']['subpath']),
                                                              gauges=self.config['input']['hydrology']['gauges']['gauge_names'],
                                                              min_pntsearch_distance=min_pntsearch_distance,
                                                              max_pntsearch_distance=max_pntsearch_distance,
                                                              sharp_edge_ratio=sharp_edge_ratio,
                                                              export_watershed_analysis=export_watershed_analysis,
                                                              output_path=self.output_dir,
                                                              boundary_stream=self.config['discretization']['meshing']['manipulation']['domain_bound']['streams'],
                                                              domain_name=self.config['info']['model_name'])
        elif self.config['input']['hydrology']['streams']['network_source']=='Hydro-EU':
            self.domain,self.streams_sorted=_delineate_basins_eu(l_vertex=self.config['discretization']['geometry']['domain'],
                                                              dem_path=os.path.join(self.config['input']['data_path'],
                                                                                    self.config['input']['topography']['dem_path']),
                                                              stream_networks=self.streams_planar,
                                                              gauge_dir=os.path.join(self.config['input']['data_path'],
                                                                                     self.config['input']['hydrology']['gauges']['subpath']),
                                                              gauges=self.config['input']['hydrology']['gauges']['gauge_names'],
                                                              sharp_edge_ratio=sharp_edge_ratio,
                                                              export_watershed_analysis=export_watershed_analysis,
                                                              output_path=self.output_dir,
                                                              boundary_stream=self.config['discretization']['meshing']['manipulation']['domain_bound']['streams'],
                                                              domain_name=self.config['info']['model_name'])
        
        
    def process_lakes(self):
        """
        
        Returns
        -------
        None.

        """
        
        self.lakes=_process_lakes(l_vertex=self.config['discretization']['geometry']['lakes'],
                                  lake_path=os.path.join(self.config['input']['data_path'],
                                                         self.config['input']['hydrology']['lakes']
                                                         ),
                                  output_path=self.output_dir)
        
        
    def process_mines(self,
                      sharp_edge_ratio=0.5):
        """
        

        Parameters
        ----------
        sharp_edge_ratio : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        """
        
        self.mines=_process_mines(l_vertex=self.config['discretization']['geometry']['mines'],
                                  sharp_edge_ratio=sharp_edge_ratio,
                                  mine_path=os.path.join(self.config['input']['data_path'],
                                                         self.config['input']['mining']['mines']),
                                  dem_path=os.path.join(self.config['input']['data_path'],
                                                        self.config['input']['topography']['dem_path']
                                                        ),
                                  output_path=self.output_dir)
        
    
    def burn_lakes(self,
                   convex_hull_buffer=10000):
        """
        

        Parameters
        ----------
        convex_hull_buffer : TYPE, optional
            DESCRIPTION. The default is 10000.

        Returns
        -------
        None.

        """
        
        self.dem_lake_burned=_burn_lakes(dem_path=os.path.join(self.config['input']['data_path'],
                                                               self.config['input']['topography']['dem_path']),
                                         convex_hull_buffer=convex_hull_buffer,
                                         output_path=None,gdf_lakes=self.lakes,
                                         gdf_watershed=self.domain)
        
        
    def map_streams(self,
                    convex_hull_buffer=10000,
                    loess_share=0.1,
                    minimum_vertices=3,
                    plot=False,
                     ):
        """
        

        Parameters
        ----------
        convex_hull_buffer : TYPE, optional
            DESCRIPTION. The default is 10000.
        loess_share : TYPE, optional
            DESCRIPTION. The default is 0.1.
        minimum_vertices : TYPE, optional
            DESCRIPTION. The default is 3.
        plot : TYPE, optional
            DESCRIPTION. The default is False.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.config['input']['hydrology']['streams']['network_source']=='Landesamt':
            
            #adress that we currently do not support different longitudinal resolutions for different stream order
            
            if len(self.config['discretization']['geometry']['streams']['longitudinal'])>1:
                print('Currently, resolution changes with stream order are not implemented')
                
            l_vertex=self.config['discretization']['geometry']['streams']['longitudinal'][0]
                       
        
            self.streams_z=_map_streams(l_vertex=l_vertex,
                                        dem_path=self.dem_lake_burned['clip_lakes'],
                                        convex_hull_buffer=convex_hull_buffer,
                                        loess_share=loess_share,
                                        minimum_vertices=minimum_vertices,
                                        stream_networks=self.streams_sorted,
                                        plot=plot,output_path=self.output_dir)
            
        elif self.config['input']['hydrology']['streams']['network_source']=='Hydro-EU':
            print('EU Hydro dataset is mapped already in monotoneous correction direction, but has bugs')
            self.streams_z = _map_streams_eu(stream_networks=self.streams_sorted,
                               gauge_dir=os.path.join(self.config['input']['data_path'],
                                   self.config['input']['hydrology']['gauges']['subpath']),
                               gauges=self.config['input']['hydrology']['gauges']['gauge_names'],
                               output_dir=self.output_dir,
                               remove_outlier_first=True,z_max=4,
                               )
                    
        
    def burn_streams(self,
                     hole_search_distance=2):
        """
        

        Parameters
        ----------

        hole_search_distance : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        #STRAHLER Order means high is bigest stream, for Landesamt it is inveres
        if self.config['input']['hydrology']['streams']['network_source']=='Landesamt':
            stream_order='reversed'
        elif self.config['input']['hydrology']['streams']['network_source']=='Hydro-EU':
            stream_order='standard'
        
        self.dem_lake_river_burned=_burn_streams(dem_path=self.dem_lake_burned['clip_lakes'],
                                                 dem_binary_path=self.dem_lake_burned['clip_lakes_binary'],
                         stream_networks=self.streams_z,lakes=self.lakes,
                         hole_search_distance=hole_search_distance,
                         dem_name=self.config['info']['model_name'],
                         output_path=self.output_dir,
                         stream_order=stream_order)


    def clip_observation_wells(self):
        """
        

        Returns
        -------
        None.

        """
        self.observation_wells=_clip_observation_wells(domain=self.domain,
                                                       observation_well_path=os.path.join(self.config['input']['data_path'],
                                                                                          self.config['input']['observations']['gw_levels']
                                                                                          ),
                                                       excluded_area_raster=os.path.join(self.config['input']['data_path'],
                                                                                         self.config['input']['geology']['aquiclude_map']
                                                                                         ),
                                                       max_welldepth=self.config['simulation']['porous_media']['aquifer_properties']['aquifer_thickness'][0],
                                                       output_path=self.output_dir,
                                                       )


    def load_gis_results(self,token='c385110a'):
        """
        Loads Data from GIS Processing output

        Returns
        -------
        None.

        """
        gis_dir=self.output_dir.replace(os.path.basename(self.output_dir),token)
        
        print('Until NOW, DEM CAN ONLY BE READ FROM FILE FOR MESHING')
        self.dem_lake_river_burned_path=os.path.join(gis_dir,'_3_mapping_burning','lausitz_clip_lakes_streams.tif')
        #get streams
        self.streams_z=dict()
        for stream in self.config['input']['hydrology']['streams']['networks']:
            self.streams_z[stream]=gpd.read_file(os.path.join(gis_dir,'_3_mapping_burning',stream+'_network_z.geojson')).set_index('reach_name',drop=True)
            
        #get lakes
        self.lakes=gpd.read_file(os.path.join(gis_dir,'_2_innersurfaces','lakes.gpkg')).set_index('Name',drop=True)

        
        #get mines
        self.mines=gpd.read_file(os.path.join(gis_dir,'_2_innersurfaces','mines.gpkg')).set_index('Name',drop=True)
        
        #the domain
        self.domain=gpd.read_file(os.path.join(gis_dir,'_1_basin_delineation','Einzugsgebiet','domain.geojson')).set_index('basin',drop=True)
        
        #the observation_wells
        self.observation_wells=gpd.read_file(os.path.join(gis_dir,'_2_innersurfaces','observation_wells.gpkg'))
        
    def load_meshing_results(self,token='c385110a'):
        """
        Loads the data from the mesh and submesh generation
        
        """
        mesh_dir=os.path.join(self.model_path,self.output_dir.replace(os.path.basename(self.output_dir),token))
        print('Load Mesh Results and Submesh Assignments from run no.: ',token)
        
        self.bulkmesh=pv.read(os.path.join(mesh_dir,'_4_mesh_generation','output','mesh',self.config['info']['model_name']+'.vtu'))
        #submeshes
        submesh_dir=os.path.join(mesh_dir,'_4_mesh_generation','output','submeshes')
        self.submeshes={name[:-4]:pv.read(os.path.join(submesh_dir,name)) for name in os.listdir(submesh_dir)}
        self.recharge=pv.read(os.path.join(mesh_dir,'_4_mesh_generation','output','recharge','recharge.vtu'))        
        #the line network
        lines=gpd.read_file(os.path.join(mesh_dir,'_4_mesh_generation','output',
                                         'line_network','mesh_adapted_streams.gpkg')
                            )
        lines=lines.set_index('reach_name',drop=True)                
        self.streams_z={'all_networks':lines}

    #%% run the entire workflow
    def gis_processing(self):
        """
        The whole gis processing in one

        Returns
        -------
        None.

        """
        self.simplify_streams()
        
        self.delineate_basin()
        
        self.process_lakes()
        
        self.process_mines()
        
        self.clip_observation_wells()

        self.burn_lakes()
        
        self.map_streams()
        
        self.burn_streams()
        
        self.gis_processed=True
        
        
    #%% run the entire Meshing Workflow
    def mesh_processing(self):
        print('Start the Meshing procedure')
        
        #check whether GIS processing is needed
        if self.config['reloads']['gis_token'] is not None:            
            print('Load GIS processing output data from run with token ',self.config['reloads']['gis_token'])            
            self.load_gis_results(token=self.config['reloads']['gis_token'])
            os.makedirs(self.output_dir,exist_ok=True)
        elif self.gis_processed==False:
            print('Run GIS PROCESSING PRIOR to MESHING')
            self.gis_processing()
            print('Until NOW, DEM DATA CAN ONLY BE READ FROM FILE FOR MESHING')
            self.dem_lake_river_burned_path=os.path.join(self.output_dir,'_3_mapping_burning','lausitz_clip_lakes_streams.tif')
            
        
        #define the datasets to be transfered to the meshing modul
        data={'streams':self.streams_z,
              'mines':self.mines,
              'lakes':self.lakes,
              'observation_wells':self.observation_wells,
              'domain':self.domain,
              'dgm_path':self.dem_lake_river_burned_path
              }
        
        #depending whether we have two 2 or 3d we use different mesh processing schemes
        if self.config['discretization']['meshing']['manipulation']['extrusion']['extrude_to_3d']:
            print('Construct 3D Meshes...')
            self.bulkmesh,self.submeshes,self.recharge,lines=_mesh_processing_3d(config=self.config,
                                                                              data=data,
                                                                              output_dir=os.path.join(self.output_dir,
                                                                                                      '_4_mesh_generation')
                                                                              )
            print('Construct 3D Meshes...done')
        else:
            print('Construct 2D Meshes...')
            self.bulkmesh,self.submeshes,self.recharge,lines=_mesh_processing_2d(config=self.config,
                                                                              data=data,
                                                                              output_dir=os.path.join(self.output_dir,
                                                                                                      '_4_mesh_generation')
                                                                              )
            print('Construct 2D Meshes...done')
        
        #if not self.holes:
        self.streams_z={'all_networks':lines}
        
        print('Meshing Procedure is finished')
        
    #%% Run the simulation with OGS
    def simulate(self):
        
        print('Start the Simulation with ',self.config['simulation']['simulator']['name'],', Currently only stationary implemented')
        
        #check whether GIS processing is needed
        if self.config['reloads']['gis_token'] is not None:            
            print('Load GIS processing output data from run with token ',self.config['reloads']['gis_token'])            
            self.load_gis_results(token=self.config['reloads']['gis_token'])
            os.makedirs(self.output_dir,exist_ok=True)
        else:
            print('Run GIS PROCESSING PRIOR to MESHING')
            self.gis_processing()
            print('Until NOW, DEM DATA CAN ONLY BE READ FROM FILE FOR MESHING')
            self.dem_lake_river_burned_path=os.path.join(self.output_dir,'_3_mapping_burning','lausitz_clip_lakes_streams.tif')
            
        #check whether mesh processing is required
        if self.config['reloads']['mesh_token'] is not None:            
            print('Load Meshing output data from run with token ',self.config['reloads']['mesh_token'])            
            self.load_meshing_results(token=self.config['reloads']['mesh_token'])
            os.makedirs(self.output_dir,exist_ok=True)
        else:
            print('RunMesh PROCESSING PRIOR to MESHING')
            self.mesh_processing()
            
        #we define the simulation directory
        sim_dir=os.path.join(self.output_dir,'_5_simulation')
        os.makedirs(sim_dir,exist_ok=True)
        
        
        #we do some time definitions
        #dependent on the time we generate different dates
        if self.config['simulation']['time']['temporal_mode'].lower()=='stationary':
            print('Prepare input for stationary run ')
            #take start time as index time
            sim_dates_index=[pd.to_datetime(self.config['simulation']['time']['start_date'],format='%Y%m%d')]
            
            #we rewrite the sim times to either 0 or 1            
            if self.config['simulation']['simulator']['name'].lower()=='opengeosys':
                sim_times=[0] #transient means one timestep only
            elif self.config['simulation']['simulator']['name'].lower()=='modflow':
                sim_times=[1]

        
        elif self.config['simulation']['time']['temporal_mode'].lower()=='transient':
            print('Prepare input for transient run ')
            #we generate the dates produced by the simulation
            step_size=int(self.config['simulation']['time']['time_end']/self.config['simulation']['time']['stress_periods'])
            
            sim_dates=[datetime.strptime(str(self.config['simulation']['time']['start_date']),'%Y%m%d')+timedelta(seconds=x) 
                       for x in range(0,self.config['simulation']['time']['time_end']+1,step_size)]
            sim_dates_index=pd.to_datetime(sim_dates)
            
            sim_times=[x for x in range(0,self.config['simulation']['time']['time_end']+1,step_size)]
        
        #next we organize the submeshes and the material properties in a way that the simulator can read it

        #%% Add boundary conditons
        
        submeshes_sim=dict({'dc':dict(),'st':dict(),'rb':dict(),'sfr':dict(),'obs':dict()})
        
        #first we update the streams
        #calculate the amount of different stream orders
        no_of_stream_orders=len(self.streams_z['all_networks']['Order'].unique())
        
        #loop through the networks
        bc_config=self.config['simulation']['boundary_conditions']
        for network in self.streams_z.values():
            for feature_name in self.submeshes.keys():
                #if is_float(feature_name[-1]):
                #    print(feature_name,' is part of a splitted polyline, correct for bc terms')
                #    name_in_network='_'.join(feature_name.split('_')[1:-1])
                #else:
                name_in_network=feature_name[3:]                        

                if name_in_network in network.index:
                    if bc_config['streams']['type'].lower()=='dirichlet':
                        submeshes_sim['dc'].update({feature_name:{'data':self.submeshes[feature_name],
                                                                  'expression':'9810*z',
                                                                  'value':None,
                                                                  'type':'Function',
                                                                  'curve':None,
                                                                  }
                                                    }
                                                   )
                    elif bc_config['streams']['type'].lower() in ['ghb','river','stream']:
                        if  bc_config['streams']['type'].lower() in ['ghb','river']:
                            bc_type_key='rb'
                        else:
                            bc_type_key='sfr'

                        
                        #first create the standard dictionary entry
                        submeshes_sim[bc_type_key].update({feature_name:{'data':self.submeshes[feature_name],
                                                                  'expression':'9810*z',
                                                                  'value':None,
                                                                  'type':'Function',
                                                                  'robin_type':bc_config['streams']['type'].lower(),
                                                                  'curve':None,
                                                                  }
                                                    }
                                                   )
                        #we loop trough the necessary additional parameters
                        for robin_parameter in ['bed_conductance','stream_cell_depth']:                            
                        
                        # We assume three different options, 
                        #first, conductance is provided for each dataset within the pandas dataframe
                        #Second, if one conductance value is given in gromola.yml, it is valid for all streams
                        #Third, the number of conductances equals the number of stream order, larger rivers get other condutancies than smaller ones
                            #raise KeyError('Not implemented yet')
                            
                            #first_case:
                            if robin_parameter in network.columns:
                                submeshes_sim[bc_type_key][feature_name].update({robin_parameter:network.loc[name_in_network,robin_parameter]})
                                
                            #second case 
                            elif len(bc_config['streams']['properties'][robin_parameter])==1 or len(bc_config['streams']['properties'][robin_parameter])!=no_of_stream_orders:
                                submeshes_sim[bc_type_key][feature_name].update({robin_parameter:bc_config['streams']['properties'][robin_parameter][0]})
                            
                            #third case 
                            elif len(bc_config['streams']['properties'][robin_parameter])==no_of_stream_orders:
                                print('Assign different ',robin_parameter, 'for each stream_order')
                                assignment_dict={i:float(bc_config['streams']['properties'][robin_parameter][i-1]) for i in self.streams_z['all_networks']['Order'].unique()}
                                network[robin_parameter]=network['Order'].copy().replace(assignment_dict)
                                submeshes_sim[bc_type_key][feature_name].update({robin_parameter:network.loc[name_in_network,robin_parameter]})
                            else:
                                raise KeyError('Please Provide Bed_Conductance data in the stream geometry file or the gromola.yml file')
                    
                            
        #update the mines
        for feature_name in self.submeshes.keys():
            if feature_name[3:] in self.mines.index:
                if bc_config['mines']['type'].lower()=='dirichlet':
                    submeshes_sim['dc'].update({feature_name:{'data':self.submeshes[feature_name],
                                                              'expression':None,
                                                              'value':float(self.mines.loc[feature_name[3:],'Sohle_NHN']),
                                                              'type':'Constant',
                                                              'curve':None,
                                                              }
                                                }
                                               )
        #update the lakes
        lake_record_times=[int(i) for i in self.lakes.columns if i.isnumeric()]
        for feature_name in self.submeshes.keys():
            if feature_name[3:] in self.lakes.index:
                if self.config['simulation']['time']['temporal_mode'].lower()=='stationary':
                    if bc_config['lakes']['type'].lower()=='dirichlet':
                        #we check whether the start_year is inside our data
                        if int(self.start_year) in lake_record_times:
                            water_level=float(self.lakes.loc[feature_name[3:],self.start_year])
                        elif int(self.start_year)>max(lake_record_times):
                            print(self.start_year,'later than last record, assume entirely flooded lake')
                            water_level=float(self.lakes.loc[feature_name[3:],'WS_Final'])
                        
                        elif int(self.start_year)<min(lake_record_times):
                            print(self.start_year,'prior to data record, take first record')
                            water_level=float(self.lakes.loc[feature_name[3:],str(min(lake_record_times))])
                            
                        #write the data    
                        
                        submeshes_sim['dc'].update({feature_name:{'data':self.submeshes[feature_name],
                                                                  'expression':None,
                                                                  'value':water_level,
                                                                  'type':'Constant',
                                                                  'curve':None,
                                                                  }
                                                    }
                                                   )
                else:
                    if bc_config['lakes']['type'].lower()=='dirichlet':
                        curve=self.lakes.loc[feature_name[3:]].to_frame()
                        curve=curve.loc[list(map(str,lake_record_times))]
                        
                        curve.set_index(pd.to_datetime(curve.index),drop=True,inplace=True)
                        curve.truncate(before=pd.to_datetime(self.config['simulation']['time']['start_date']))
                        curve['dtime']=curve.index
                        curve['dtime']=(curve['dtime'].diff()//np.timedelta64(1,'s')).fillna(0).astype(int).cumsum()
                        curve=curve.reset_index(drop=True).set_index('dtime')[feature_name[3:]]
                        
                        submeshes_sim['dc'].update({feature_name:{'data':self.submeshes[feature_name],
                                                                  'expression':None,
                                                                  'value':1,
                                                                  'type':'Constant',
                                                                  'curve':curve,
                                                                  }
                                                    }
                                                   )
                        
                    
        #%%add source terms, in our case the recharge
        if self.config['simulation']['source_terms']['recharge']['recharge_type'].lower()=='curve_array':
            if self.config['input']['hydrology']['recharge']['recharge_curve'] is not None:
                print('load recharge curve from ',self.config['input']['hydrology']['recharge']['recharge_curve'])
                curve=pd.read_csv(os.path.join(self.config['input']['data_path'],
                                               self.config['input']['hydrology']['recharge']['recharge_curve']),
                                  index_col='time',parse_dates=True)
                
                curve=curve.reindex(curve.index.union(sim_dates_index)).interpolate(method='time').reindex(sim_dates_index)
                
                #reindex with time steps in seconds
                curve.index=sim_times
                
                submeshes_sim['st'].update({'recharge':{'data':self.recharge,'type':'MeshElement',
                                  'field_name':'recharge','curve':curve,
                                  }
                                            }
                                       )
            else:
                raise KeyError('Provide Valid Recharge curve path if this options should be used')
        elif self.config['simulation']['source_terms']['recharge']['recharge_type'].lower()=='constant_array':
            print('assume recharge to be a constant array')
            submeshes_sim['st'].update({'recharge':{'data':self.recharge,'type':'MeshElement',
                              'field_name':'recharge','curve':None,
                              }
                                        }
                                   )
        else:
            raise KeyError('Currently only curve_array and constant_array are implemented options for the recharge')

        
        
        #%%add information don observation wells
    
        #for stationary wells, not much need to be changed
        if self.observation_wells['transient'].iloc[0]==0:
            self.observation_wells.index.name='time'
            if len(sim_times)==1:
                self.observation_wells.index=sim_times*len(self.observation_wells)

                #raise KeyError('Select transient observation well dataset for transient simulations')
            self.observation_wells=self.observation_wells[['observation','well_id']].dropna()
        elif self.observation_wells['transient'].iloc[0]==1:
            #for transient wells we have to assign the time in according o the simulation timesteps
            self.observation_wells['time']=pd.to_datetime(self.observation_wells['Messzeitpunkt'])
            self.observation_wells.set_index('time',drop=True,inplace=True)
            self.observation_wells=self.observation_wells[['observation','well_id','MKZ']]                
            
            #reindex by time
            self.observation_wells=self.observation_wells.groupby('MKZ').apply(lambda x:_time_reindexing(x,sim_dates_index,sim_times))
            self.observation_wells=self.observation_wells[['observation','well_id']].dropna()
            self.observation_wells=self.observation_wells.reset_index().set_index('time')
        
        #reduce to relevant columns
        self.observation_wells=self.observation_wells[['observation','well_id']].dropna()
            
        #add to observation term data
        submeshes_sim['obs'].update({'observation_wells':{'data':self.submeshes['observation_wells'],
                                    'type':'pressure','source_data':self.observation_wells}
                                     }
                                   )
                
            
            
                        
                
        #%%add fluxes
     
        #we generate the dates produced by the simulation
        # load flux source_data
        flux_source_data=pd.read_csv(os.path.join(self.config['input']['data_path'],
                                                  self.config['input']['observations']['gw_extraction']),
                                     delimiter=';',index_col='Jahr')        
        flux_source_data.index=pd.to_datetime(flux_source_data.index,format='%Y')
        flux_source_data=flux_source_data.fillna(method='backfill').fillna(method='ffill')

        flux_source_data=flux_source_data.reindex(flux_source_data.index.union(sim_dates_index)).interpolate(method='time').reindex(sim_dates_index)

        #reindex with time steps in seconds
        flux_source_data.index=sim_times        
        # get all features        
        if self.config['simulation']['time']['flux_output_features'][0] is not None:
            flux_features_all=deepcopy(self.config['simulation']['time']['flux_output_features'])
            flux_features_all.extend(flux_source_data.columns)
            flux_features_all=set(flux_features_all)
        else:
            flux_features_all=flux_source_data.columns
        for flux_feature in flux_features_all:
            #look for all submehes which contain the feature name from the flux observation list
            #can be multiple values due to the splitting of river features
            flux_submeshes=[feature for feature in self.submeshes if flux_feature in feature]
            if len(flux_submeshes)==0:
                print(flux_feature, 'is either spelled incorectly or not part of any boundary feature',end='')
                print('Currently only geometrys provided in the input data can be used for flux estimation')
                continue
            else:
                #write down the submeshes
                for flux_submesh in flux_submeshes:
                    #if observation data for the fluxes is available we add it to data otherwise not
                    if flux_submesh.split('_')[-2]+'_'+flux_submesh.split('_')[-1] in flux_source_data.columns:
                        source_data=flux_source_data.loc[:,flux_submesh.split('_')[-2]+'_'+flux_submesh.split('_')[-1]]
                    else:
                        source_data=None
                    #flux_submesh_data=[submeshes_sim[key][flux_submesh]['data'] for key in submeshes_sim.keys() if flux_submesh in submeshes_sim[key].keys()]
                    submeshes_sim['obs'].update({flux_submesh:{'data':self.submeshes[flux_submesh],
                                                    'type':'flux','source_data':source_data}
                                                     }
                                                   )
                        
                            
        #we further load the main flux file
        if self.config['info']['model_name']+'_boundary_fluxes' in self.submeshes.keys() and self.config['simulation']['simulator']['name']=='opengeosys':
            flux_mesh=self.submeshes[self.config['info']['model_name']+'_boundary_fluxes']
            if self.config['simulation']['porous_media']['media_type'].lower()=='homogeneous':
                if self.config['discretization']['meshing']['manipulation']['hydraulic_barriers']['barrier_cells_as_holes'] is False:
                    flux_mesh['MaterialIDs'][flux_mesh['MaterialIDs']<max(self.bulkmesh['MaterialIDs'])]=0
                    flux_mesh['MaterialIDs'][flux_mesh['MaterialIDs']>0]=1
                else:                
                    flux_mesh['MaterialIDs']=flux_mesh.n_cells*[0]
                
            
            print('Append bc fluxes submesh to observations for opengeosys run')
            submeshes_sim['obs'].update({self.config['info']['model_name']+'_boundary_fluxes':{'data':flux_mesh,
                                        'type':'boundary_fluxes','source_data':None}
                                         }
                                       )
            
            
                    
            


        #form the material group dict
        porous_config=deepcopy(self.config['simulation']['porous_media']['aquifer_properties'])
        storages = porous_config['storage']
        print('storage is ', storages)
        permeabilities = porous_config['permeability']
        porosities = porous_config['porosity']
        aquifer_thickness=porous_config['aquifer_thickness']
        #number of different material groups
        if self.config['simulation']['porous_media']['media_type'].lower()=='homogeneous':
            mat_counts=1
            print('Media is homogenous, we homogenize matids information by one group only')
            if self.config['input']['mining']['hydraulic_barriers'] is not None and self.config['discretization']['meshing']['manipulation']['hydraulic_barriers']['barrier_cells_as_holes'] is False:
                self.bulkmesh['MaterialIDs'][self.bulkmesh['MaterialIDs']<max(self.bulkmesh['MaterialIDs'])]=0
                self.bulkmesh['MaterialIDs'][self.bulkmesh['MaterialIDs']>0]=1
            else:                
                self.bulkmesh['MaterialIDs']=self.bulkmesh.n_cells*[0]

            
        elif self.config['simulation']['porous_media']['media_type'].lower()=='heterogeneous':
            
            print('Heterogeneous conditions assumed, make sure that mat property list length is same as number of material groups')
            mat_counts=len(np.unique(self.bulkmesh['MaterialIDs']))
            
            if self.config['input']['mining']['hydraulic_barriers'] is not None and self.config['discretization']['meshing']['manipulation']['hydraulic_barriers']['barrier_cells_as_holes'] is False:
                mat_counts=mat_counts-1
            
            print(str(mat_counts),' different material groups detected')
            #fix if less than material groups
            if len(storages)!=mat_counts:
                print('Provided Storage parameters unequal than material groups, take first value')
                storages=[storages[0]]*mat_counts
            if len(permeabilities)!=mat_counts:
                print('Provided permeabilities parameters unequal than material groups, take first value')
                permeabilities=[permeabilities[0]]*mat_counts
            if len(porosities)!=mat_counts:
                print('Provided porosities parameters unequal than material groups, take first value')
                porosities=[porosities[0]]*mat_counts
                
            if len(aquifer_thickness)!=mat_counts:
                print('Provided aquifer_thickness parameters unequal than material groups, take first value')
                aquifer_thickness=[aquifer_thickness[0]]*mat_counts
        
        # in case that we have the barriers included we add one material group
        if self.config['input']['mining']['hydraulic_barriers'] is not None and self.config['discretization']['meshing']['manipulation']['hydraulic_barriers']['barrier_cells_as_holes'] is False:
            print('Add a material group to adress for the barriers')
            mat_counts+=1
            storages.append(storages[0])
            porosities.append(porosities[0])
            permeabilities.append(porous_config['barrier_permeability'])
            aquifer_thickness.append(aquifer_thickness[0])
            
        

                              
        #load the relevant model class
        if self.config['simulation']['simulator']['name'].lower()=='opengeosys':
            
            #form the dictionary
            mat_props=dict()
            if self.config['discretization']['meshing']['manipulation']['extrusion']['extrude_to_3d']:
                print('Compute Transmissivity using aquifer thickness')
            else:
                print('We have 3D model, no need for explicit aquifer thickness')
                aquifer_thickness=list(map(lambda x:1,aquifer_thickness)) # make all to one
            
            for mat_id in range(0,mat_counts):
                mat_props.update({mat_id:{'porosity':porosities[mat_id],
                                          'storage':storages[mat_id],
                                          'permeability':permeabilities[mat_id]*aquifer_thickness[mat_id],
                                          'reference_temperature':self.config['simulation']['porous_media']['fluid_properties']['temperature'],
                                          'viscosity':self.config['simulation']['porous_media']['fluid_properties']['viscosity'],
                                          'density':self.config['simulation']['porous_media']['fluid_properties']['density']
                                          }
                                  }
                                )                
            #simulate                                 
            #sim_time=tuple((float(self.time_start),float(self.time_end),int(self.stress_periods),int(self.output_interval))),
            model=_ogs_simulator(simulation_dir=sim_dir,
                                 project_name=self.config['info']['model_name'],
                                 submeshes=submeshes_sim,
                                 initial_head=self.config['simulation']['initial_conditions']['initial_head'],
                                 mesh=self.bulkmesh,
                                 sim_time={'time_start':self.config['simulation']['time']['time_start'],
                                           'time_end':self.config['simulation']['time']['time_end'],
                                           'stress_periods':int(self.config['simulation']['time']['stress_periods']),
                                           'timesteps_per_spd':int(self.config['simulation']['time']['time_step_per_spd']),
                                           'output_interval':int(self.config['simulation']['time']['output_interval']),
                                                                            },
                                 aquifer_type=self.config['simulation']['porous_media']['aquifer_type'],
                                 temporal_mode=self.config['simulation']['time']['temporal_mode'],
                                 numerics=self.config['simulation']['numerics']['option'],
                                 material_properties=mat_props,
                                 ogs_path= self.config['simulation']['simulator']['path'],
                                 transient_metrics=self.config['calibration']['metrics'],
                                 numerical_precision=self.config['simulation']['numerics']['numerical_precision'],
                                 )
            
            
        elif self.config['simulation']['simulator']['name'].lower()=='modflow':
            
            #correct the output parameters depending on the calibration parameter, we need budget for fluxes
            if self.config['calibration']['type_of_observation'].lower()=='flux' or self.config['simulation']['time']['flux_output_features'][0] is not None:
                output_parameters=['head','budget']
            else:
                output_parameters=['head']
            
            #form the dictionary
            mat_props=dict()
            for mat_id in range(0,mat_counts):
                mat_props.update({mat_id:{'porosity':porosities[mat_id],
                                          'storage':storages[mat_id],
                                          'permeability':permeabilities[mat_id],
                                          'reference_temperature':self.config['simulation']['porous_media']['fluid_properties']['temperature'],
                                          'viscosity':self.config['simulation']['porous_media']['fluid_properties']['viscosity'],
                                          'density':self.config['simulation']['porous_media']['fluid_properties']['density']
                                          }
                                  }
                                )
                            
            model=_modflow_simulator(sim_dir=sim_dir,
                                 project_name=self.config['info']['model_name'],
                                 submeshes=submeshes_sim,
                                 initial_head=self.config['simulation']['initial_conditions']['initial_head'],
                                 mesh=self.bulkmesh,
                                 time={'time_start':self.config['simulation']['time']['time_start'],
                                       'date_start':str(self.config['simulation']['time']['start_date']),
                                       'time_end':self.config['simulation']['time']['time_end'],
                                       'stress_periods':int(self.config['simulation']['time']['stress_periods']),
                                       'timesteps_per_spd':int(self.config['simulation']['time']['time_step_per_spd']),
                                       'output_interval':int(self.config['simulation']['time']['output_interval']),
                                        },                                           
                                 aquifer_type=self.config['simulation']['porous_media']['aquifer_type'],
                                 temporal_mode=self.config['simulation']['time']['temporal_mode'],
                                 numerics=self.config['simulation']['numerics']['option'],
                                 material_properties=mat_props,
                                 aquifer_thickness=aquifer_thickness[0],
                                 mf6ExeName= self.config['simulation']['simulator']['path'],
                                 start_date_time =self.config['simulation']['time']['start_date'],
                                 transient_metrics=self.config['calibration']['metrics'],
                                 output_datatype=self.config['simulation']['time']['output_datatype'],
                                 reload_packages=True,
                                 output_parameters=output_parameters,
                                 exlusion_area_path=os.path.join(self.config['input']['data_path'],self.config['input']['geology']['aquiclude_map']),
                                 numerical_precision=self.config['simulation']['numerics']['numerical_precision']
                                 )
        else:
            raise KeyError('Only MODFLOW and OGS supported')
                
            #run the simulation
        model.simulate()
            
        return model
            
    #%% Calibrate the model
    """
    First we run the model once,
    We set the simulation token
    We create the pest files
    We run the model in the pest_loop
    """
    def calibrate_model(self,keep_mesh=True,keep_geometry=True):
        print('Run Simulation in Calibration Mode')
        #first we simulate one time
        self.simulate()
        #decide whether we use the geometry and mesh from first simulation or not
        if self.config['reloads']['gis_token'] is None:
            gis_token_initial=self.identifier
            self.config['reloads']['gis_token']=gis_token_initial
        if self.config['reloads']['mesh_token'] is None:
            mesh_token_initial=self.identifier
            self.config['reloads']['mesh_token']=mesh_token_initial
        #set the simulation token
        self.config['reloads']['simulation_token']=self.identifier
        
        #depending on the initial conditons we rewrite the tokens        

        
        
        
        #%% define which observation data should be used for calibration
        if self.config['calibration']['type_of_observation']=='head':
            print('Use hydraulic head observation as calibration target')
            simulation_results_path=os.path.join(self.output_dir,'_5_simulation','output_pst_head_metrics.csv')
        elif self.config['calibration']['type_of_observation']=='flux':
            print('use Fluxes from flux observation list as calibration target')
            simulation_results_path=os.path.join(self.output_dir,'_5_simulation','output_pst_fluxes_metrics.csv')
        else:
            raise KeyError('Select either flux or head as calibration target')
        
        #write out the config with the new simulation token
        with open(os.path.join(self.output_dir,'gromola.yml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)
            
        print('Generate the Pest files')
        calibration_dir=os.path.join(self.output_dir,'_6_calibration')
        os.makedirs(calibration_dir,exist_ok=True)
        
        if self.config['calibration']['important_calibration_features'][0] is not None and self.config['calibration']['important_calibration_features_factor'][0] is not None:
        
            #bring the factor and feature names together
            
            if len(self.config['calibration']['important_calibration_features_factor'])!=len(self.config['calibration']['important_calibration_features']):
                print('important_calibration_features_factor length is unequal to length of important_calibration_features, take first value')
                important_calibration_features_factors=[self.config['calibration']['important_calibration_features_factor'][0]]*len(self.config['calibration']['important_calibration_features'])
            else:
                important_calibration_features_factors=self.config['calibration']['important_calibration_features_factor']
            #we merge to dictionary
            important_calibration_features=dict(zip(self.config['calibration']['important_calibration_features'],important_calibration_features_factors))
        else:
            important_calibration_features=dict()
        
        pest_model=_pest(output_dir=calibration_dir,
                        pest_exe_path=os.path.normpath(self.config['calibration']['calibrator_path']),
                        pest_config_path=os.path.join(self.output_dir,'calibration_config.csv'),
                        model_config_path=os.path.join(self.output_dir,'gromola.yml'),
                        simulation_results_path=simulation_results_path,
                        pest_filename=self.config['info']['model_name'],
                        model_dir=os.path.join(self.model_path,'output',self.identifier),
                        important_calibration_features=important_calibration_features,
                     )
        pest_model.generate_pest_files()
        
        #we run the pest model
        print('Run Calibration with Pest...',end='')
        pest_path=os.path.normpath(self.config['calibration']['calibrator_path'])
        os.chdir(calibration_dir)
        os.system(pest_path+' '+os.path.join(calibration_dir,self.config['info']['model_name']+'.pst')+' >log.log')
        print('...done')
        
        #remove the simulation token
        self.config['reloads']['simulation_token']=''
        
        #depending on the initial conditons we again rewrite the tokens
        """
        if gis_token_initial is None:
            self.config['reloads']['gis_token']=''
        if gis_token_initial is None:
            self.config['reloads']['mesh_token']=''
        """
        #write out the config with the new simulation token
        with open(os.path.join(self.output_dir,'gromola.yml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)
            
        #change directory back to normal working directory
        os.chdir(self.model_path)
    
    
        
       
            
        
        
    

        
        
        
            
 