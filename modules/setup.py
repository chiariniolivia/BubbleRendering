import mitsuba as mi
import drjit as dr

import numpy as np

import time
import pathlib

# Directories
scene_dir = './chamber_model/components/'
object_format = '.ply'
output_dir = './outputs/'
input_dir = './chamber_images/'

# Select the best-available Mitsuba variant
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb', 'scalar_rgb')

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array"""
    return arr/np.linalg.norm(arr)

def to_mono(image_rgb: np.ndarray | mi.TensorXf) -> np.ndarray:
    """Converts the image to single channel by taking average of all color channels. 
    Warning! This returns a Numpy array, not a mi.TensorXf so this cannot be part of the AD graph.
    To convert to mono for use in AD, it is suggested to select one color channel instead."""
    if (len(image_rgb.shape) != 3):
        print(f"Image provided to to_mono has {len(image_rgb.shape)} dimensions! Expected 3 dimensions.")
        return -1
    return np.average(image_rgb, axis=2)

def to_uint8(image: np.ndarray | mi.TensorXf) -> np.ndarray:
    return (image * 255.9999).astype(np.uint8)

def create_materials(with_fluids: bool = False, fermi_chamber:bool = True, iors: dict = {
    'sapphire_ior': 1.77,
    'quartz_ior': 1.458,
    'CF4_ior': 1.22,
    'Ar_ior': 1.17,
    'vacuum_ior': 1.0,
}) -> dict:
    """
    Define material properties depending on whether the chamber is filled with fluid.
    """
    sapphire_ior = iors['sapphire_ior']
    quartz_ior = iors['quartz_ior']
    CF4_ior = iors['CF4_ior']
    Ar_ior = iors['Ar_ior']
    vacuum_ior = iors['vacuum_ior']

    materials = dict()

    if with_fluids:
        materials['sapphire_outer'] = {
            'type': 'dielectric',
            'int_ior': sapphire_ior,
            'ext_ior': vacuum_ior,
        }
        materials['sapphire_inner'] = {
            'type': 'dielectric',
            'int_ior': sapphire_ior,
            'ext_ior': CF4_ior,      
        }
        materials['outer_jar_outer_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': CF4_ior, 
        }
        materials['outer_jar_inner_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': Ar_ior,
        }
        materials['inner_jar_outer_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': Ar_ior,
        }
        materials['inner_jar_inner_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': vacuum_ior,
        }
        materials['bubble_mat'] = {
            'type': 'dielectric',
            'int_ior': vacuum_ior,
            'ext_ior': Ar_ior,
        }
    else:
        materials['sapphire_outer'] = {
            'type': 'dielectric',
            'int_ior': sapphire_ior,
            'ext_ior': vacuum_ior,
        }
        materials['sapphire_inner'] = {
            'type': 'dielectric',
            'int_ior': sapphire_ior,
            'ext_ior': vacuum_ior,      
        }
        materials['outer_jar_outer_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': vacuum_ior, 
        }
        materials['outer_jar_inner_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': vacuum_ior,
        }
        materials['inner_jar_outer_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': vacuum_ior,
        }
        materials['inner_jar_inner_surface_mat'] = {
            'type': 'dielectric',
            'int_ior': quartz_ior,
            'ext_ior': vacuum_ior,
        }
        materials['bubble_mat'] = {
            'type': 'dielectric',
            'int_ior': vacuum_ior,
            'ext_ior': vacuum_ior,
        }

    materials['ptfe'] = { 
        'type': 'roughplastic',
        'distribution': 'beckmann',
        'alpha': 0.05, # roughness
        'nonlinear': True,
        'diffuse_reflectance': {'type': 'rgb', 'value': 0.4},
    }

    materials['pcb_mat'] = {
        'type': 'roughplastic',
        'distribution': 'beckmann',
        'alpha': 0.8, # roughness
        'nonlinear': True,
        'diffuse_reflectance': {'type': 'rgb', 'value': [4/255, 99/255, 7/255]}, # pcb green
    }

    materials['resin'] = {
        'type': 'roughdielectric',
        'distribution': 'beckmann',
        'alpha': 0.2, # roughness
        'int_ior': 'bk7', # borosilicate glass (approx)
        'ext_ior': 'air'
    }

    # The true materials for the pressure vessel and fill line is stainless steel (AISI 304)
    # Unfortunately, since mitsuba does not have a preset for it, we are approximating with this (for now)
    materials['rough_test'] = {
        'type': 'roughconductor',
        'material': 'Al',
        'specular_reflectance': {
            'type': 'spectrum',
            'value': 0.3,
        },
        'alpha': 0.1, # roughness
    }

    materials['smooth_test'] = {
        'type': 'conductor',
        'material': 'Al',
        'specular_reflectance': {
            'type': 'spectrum',
            'value': 0.3,
        },
    }

    materials['copper'] = {
        'type': 'roughconductor',
        'material': 'Cu',
        'alpha': 0.1,
    }

    materials['black'] = {
        'type': 'diffuse',
        'id': 'black-bsdf',
        'reflectance': { 'type': 'spectrum', 'value': 0 },
    }

    materials['black_plastic'] = {
        'type': 'roughplastic',
        'distribution': 'beckmann',
        'alpha': 0.1, # roughness
        'nonlinear': True,
        'diffuse_reflectance': {'type': 'rgb', 'value': 0.05},
    }

    # Since I was too lazy to appropriately segment LEDs in the SNOLAB chamber, 
    # we just make the emmitance lower to compensate for larger LEDs -- though this
    # does result in 'blocky' reflections of the LEDs
    if fermi_chamber:
        materials['emitter'] = {
            'type':'area',
            'radiance': {
                'type': 'spectrum',
                'value': 60.0
            },
        }
    else:
        materials['emitter'] = {
            'type':'area',
            'radiance': {
                'type': 'spectrum',
                'value': 30.0
            },
        }
    
    return materials


def load_components(materials: dict, use_distorted_jar: bool = True, fermi_chamber: bool = True) -> dict:
    """
    Load chamber components and apply pre-defined materials depending on chamber variant.
    WARNING: Will raise an error if a required material is not defined.
    """
    components = dict()

    components['cables'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'cables'+object_format,
        'face_normals': False,
        'bsdf': materials['smooth_test'],
    })

    components['fiducials'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'fiducials'+object_format,
        'face_normals': True, # Don't want to smooth the edges here
        'bsdf': materials['resin'],
    })

    components['fill_line'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'fill_line'+object_format,
        'face_normals': True,
        'bsdf': materials['smooth_test'],
    })

    components['inner_tower'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'inner_tower'+object_format,
        'face_normals': True,
        'bsdf': materials['ptfe'],
    })

    components['reflector_supports'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'reflector_supports'+object_format,
        'face_normals': True, # default computed face normals cause artifacts...
        'bsdf': materials['copper'],
    })

    components['jar_reflectors'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'jar_reflectors'+object_format,
        'face_normals': True,
        'bsdf': materials['ptfe'],
    })

    components['jar_reflector_screws'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'jar_reflector_screws'+object_format,
        'face_normals': False,
        'bsdf': materials['rough_test'],
    })

    components['sipm_covers'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'sipm_covers'+object_format,
        'face_normals': False,
        'bsdf': materials['black_plastic'],
    })

    components['sipm_detectors'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'sipm_detectors'+object_format,
        'face_normals': False,
        'bsdf': materials['black'],
    })

    components['sipm_supports'] = mi.load_dict({
        'type': 'ply',
        'filename': scene_dir+'sipm_supports'+object_format,
        'face_normals': False,
        'bsdf': materials['ptfe'],
    })

    if use_distorted_jar:
        # OUTER JAR
        components['outer_jar_outer_surface_top'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'outer_jar_outer_surface_top_distorted'+object_format,
                'face_normals': False,
                'bsdf': materials['outer_jar_outer_surface_mat'],
        })
        components['outer_jar_outer_surface_bottom'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'outer_jar_outer_surface_bottom'+object_format,
                'face_normals': False,
                'bsdf': materials['outer_jar_outer_surface_mat'],
        })
        components['outer_jar_inner_surface_top'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'outer_jar_inner_surface_top_distorted'+object_format,
                'face_normals': False,
                'bsdf': materials['outer_jar_inner_surface_mat'],
        })
        components['outer_jar_inner_surface_bottom'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'outer_jar_inner_surface_bottom'+object_format,
                'face_normals': False,
                'bsdf': materials['outer_jar_inner_surface_mat'],
        })
        # INNER JAR
        components['inner_jar_outer_surface_top'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'inner_jar_outer_surface_top_distorted'+object_format,
                'face_normals': False,
                'bsdf': materials['inner_jar_outer_surface_mat'],
        })
        components['inner_jar_outer_surface_bottom'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'inner_jar_outer_surface_bottom'+object_format,
                'face_normals': False,
                'bsdf': materials['inner_jar_outer_surface_mat'],
        })
        components['inner_jar_inner_surface_top'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'inner_jar_inner_surface_top_distorted'+object_format,
                'face_normals': False,
                'bsdf': materials['inner_jar_inner_surface_mat'],
        })
        components['inner_jar_inner_surface_bottom'] = mi.load_dict({
                'type': 'ply',
                'filename': scene_dir+'inner_jar_inner_surface_bottom'+object_format,
                'face_normals': False,
                'bsdf': materials['inner_jar_inner_surface_mat'],
        })
    else:
        components['outer_jar_outer_surface'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'outer_jar_outer_surface'+object_format,
            'face_normals': False,
            'bsdf': materials['outer_jar_outer_surface_mat'],
        })
        components['outer_jar_inner_surface'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'outer_jar_inner_surface'+object_format,
            'face_normals': False,
            'bsdf': materials['outer_jar_inner_surface_mat'],
        })
        components['inner_jar_outer_surface'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'inner_jar_outer_surface'+object_format,
            'face_normals': False,
            'bsdf': materials['inner_jar_outer_surface_mat'],
        })
        components['inner_jar_inner_surface'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'inner_jar_inner_surface'+object_format,
            'face_normals': False,
            'bsdf': materials['inner_jar_inner_surface_mat'],
        })

    if fermi_chamber:
        components['led_backings'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_backings'+object_format,
            'face_normals': True,
            'bsdf': materials['ptfe'],
        })

        components['led_ring_1'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_ring_1'+object_format,
            'face_normals': True,
            'bsdf': materials['black'],
            'focused-emitter': materials['emitter'],
        })

        components['led_ring_2'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_ring_2'+object_format,
            'face_normals': True,
            'bsdf': materials['black'],
            'focused-emitter': materials['emitter'],
        })

        components['led_ring_3'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_ring_3'+object_format,
            'face_normals': True,
            'bsdf': materials['black'],
            'focused-emitter': materials['emitter'],
        })

        components['pcbs'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'pcbs'+object_format,
            'face_normals': True,
            'bsdf': materials['pcb_mat'],
        })
        
        components['pressure_vessel'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'pressure_vessel'+object_format,
            'face_normals': True,
            'bsdf': materials['rough_test'],
        })

        components['dome_reflectors'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'dome_reflectors'+object_format,
            'face_normals': True,
            'bsdf': materials['ptfe'],
        })

        # NOTE: for snolab, dome reflector screws are part of PV
        components['dome_reflector_screws'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'dome_reflector_screws'+object_format,
            'face_normals': False,
            'bsdf': materials['rough_test'],
        })

        components['viewports_outer'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'sapphire_viewports_outer_surface'+object_format,
            'face_normals': True,
            'bsdf': materials['sapphire_outer'],
        })

        components['viewports_inner'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'sapphire_viewports_inner_surface'+object_format,
            'face_normals': True,
            'bsdf': materials['sapphire_inner'],
        })

    else: # SNOLAB Chamber components
        components['led_ring_1'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_ring_1_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['black'],
            'focused-emitter': materials['emitter'],
        })

        components['led_ring_2'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_ring_2_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['black'],
            'focused-emitter': materials['emitter'],
        })

        components['led_ring_3'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'led_ring_3_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['black'],
            'focused-emitter': materials['emitter'],
        })

        components['pcbs'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'pcbs_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['pcb_mat'],
        })

        components['pressure_vessel'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'pressure_vessel_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['rough_test'],
        })

        components['dome_reflectors'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'dome_reflectors_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['ptfe'],
        })

        components['viewports_outer'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'sapphire_viewports_outer_surface_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['sapphire_outer'],
        })

        components['viewports_inner'] = mi.load_dict({
            'type': 'ply',
            'filename': scene_dir+'sapphire_viewports_inner_surface_snolab'+object_format,
            'face_normals': True,
            'bsdf': materials['sapphire_inner'],
        })

    return components


def create_sensor(sensor_number=2, sample_count=1024, fov=100, width=1280, height=800, 
                fermi_chamber=True, to_world=None, type='perspective', near_clip=0.01, 
                far_clip=1000.0, film:dict=None, sampler:dict=None, dist_from_viewport:float=1.0) -> dict:
    """
    Creates a sensor based on provided args.
    Note: if 'film' is provided, the 'width' and 'height' will not be used.
          if 'sampler' is provided, then 'sample_count' will not be used.
          if 'to_world' is provided, 'sensor_number' will not be used.
    """
    if film is None:
        film = {
            'type': 'hdrfilm',
            'rfilter': {
                # 'type': 'gaussian'
                'type': 'box'
            },
            'width': width,
            'height': height,
        }
    
    if sampler is None:
        sampler = {
            'type': 'multijitter',
            # 'type': 'independent',
            'sample_count': sample_count
        }

    if to_world is None:
        try:
            if sensor_number == 1 and fermi_chamber: # Fermi cam 1
                o = np.array([-5.772, -9.9973, 16.925]) # viewport outer face center
                n = normalize(np.array([0.1913462, 0.33140403, -0.92388207])) # viewport normal
                c_o = o - dist_from_viewport*n # camera origin
                t = o + n # camera target
                to_world = mi.ScalarTransform4f.look_at(origin=c_o, target=t, up=normalize([-1, 1, 0]))
            elif sensor_number == 2 and fermi_chamber: # Fermi cam 2
                o = np.array([11.546, 0.0, 16.926])
                n = normalize(np.array([-3.82694870e-01, 2.20595775e-05, -9.23874795e-01]))
                c_o = o - dist_from_viewport*n
                t = o + n
                to_world = mi.ScalarTransform4f.look_at(origin=c_o, target=t, up=[0, -1, 0])
            elif sensor_number == 3 and fermi_chamber: # Fermi cam 3
                o = np.array([-5.7725, 9.9982, 16.928])
                n = normalize(np.array([0.19134682, -0.33141729, -0.92387718]))
                c_o = o - dist_from_viewport*n
                t = o + n
                to_world = mi.ScalarTransform4f.look_at(origin=c_o, target=t, up=normalize([1, 1, 0]))
            elif sensor_number == 1 and not fermi_chamber: # SNOLAB cam 1
                to_world = mi.ScalarTransform4f.look_at(origin=[-7.0, -11.57, 19.13], target=[14, 14, -65], up=normalize([-1, 1, 0]))
            elif sensor_number == 2 and not fermi_chamber: # SNOLAB cam 2
                to_world = mi.ScalarTransform4f.look_at(origin=[13.40, 0.0, 19.13], target=[-20, 0, -65], up=[0, -1, 0])
            elif sensor_number == 3 and not fermi_chamber: # SNOLAB cam 3
                to_world = mi.ScalarTransform4f.look_at(origin=[-7.0, 11.57, 19.13], target=[14, -14, -65], up=normalize([1, 1, 0]))
            else:
                raise Exception('No valid sensor_number given!')
        except Exception as e:
            print(e)

    sensor = {
        'type': type,
        'near_clip': near_clip,
        'far_clip': far_clip,
        'fov': fov,
        'film': film,
        'sampler': sampler,
        'to_world': to_world,
    }

    return sensor


def load_scene(components:dict=None, sensor:dict=None, integrator:dict={'type': 'path'}) -> mi.Object:
    """
    Compiles scene from components (or creates scene if none is provided) and loads them to the GPU.
    """
    if components is None:
        components = load_components(materials=create_materials())

    if sensor is None: 
        sensor = create_sensor()

    scene_dict = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': sensor,
    }
    scene_dict.update(components)

    # Load scene to GPU
    scene = mi.load_dict(scene_dict)
    return scene


def render(scene=None, denoise=True, save_mono=False, save_rgb=False, save_string_append="") -> mi.TensorXf:
    """
    Render the scene with provided keyword args.
    Note: use create_materials(), load_components(), make_sensor() to create the scene. 
    These are excluded from this function to allow editing those parameters before passing
    them in to render and to allow for Mitsuba optimization.

    Args:
        scene: the Mitsuba scene with all parameters already set for rendering.
            Leave as `None` to create default scene.
        denoise: set True to denoise the image using NVIDIA Optix -- requires a cuda enabled gpu
        save_mono: set True to save the image as a mono-color png image.
        save_rgb: set True to save the image as a rgb-color png image.
        save_string_append: appends custom string to end of image name when saving
            Note: the default image name is render_time + 'rgb' or 'mono' + '.png'

    Returns:
        ONLY the rgb image! User should use `to_mono(image)` to convert to mono.
    """
    if scene is None:
        # Create default scene
        scene = load_scene()

    image_rgb = mi.render(scene)

    if denoise:
        denoiser = mi.OptixDenoiser(input_size=(image_rgb.shape[1], image_rgb.shape[0]))
        image_rgb = denoiser(image_rgb)
    
    if save_mono or save_rgb:
        render_date = time.strftime('%Y_%m_%d')
        render_time = time.strftime('%H_%M_%S')
    
        # Create new directory for today if it does not already exist
        pathlib.Path(output_dir+render_date).mkdir(parents=True, exist_ok=True)
        image_name = output_dir+render_date+'/'+render_time

        # This is so there is not a trailing underscore when save_string_append is empty
        if save_string_append != "":
            save_string_append = "_"+save_string_append
        
        if save_mono:
            image_name_mono = image_name + '_mono' + save_string_append + '.png'
            image_mono = to_mono(image_rgb)
            mi.util.write_bitmap(image_name_mono, image_mono)
            print('Saved Image:', image_name_mono)
        if save_rgb:
            image_name_rgb = image_name + '_rgb' + save_string_append + '.png'
            mi.util.write_bitmap(image_name_rgb, image_rgb)
            print('Saved Image:', image_name_rgb)

    return image_rgb