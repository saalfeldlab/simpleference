import json
import os

def configure(**kwargs):
    config = kwargs
    print(config.keys())
    if 'experiment_name' not in config:
        config['experiment_name']      = 'baseline_DTU2'
    if 'sample' not in config:
        config['sample']               = 'A'
    if 'iteration' not in config:
        config['iteration']            = 90000
    if 'output_shape' not in config:
        config['output_shape']         = (71, 650, 650)
    if 'output_base' not in config:
        config['output_base']          = '/nrs/saalfeld/heinrichl/synapses/miccai_experiments/'
    if 'net_io_input' not in config:
        config['net_io_input'] = 'raw'
    if 'net_io_output' not in config:
        config['net_io_output'] = 'dist'

    # independent arguments
    if 'data_key' not in config:
        config['data_key']             = 'volumes/raw'
    if 'mask_keys' not in config:
        config['mask_keys']            = ('volumes/masks/validation', 'volumes/masks/training')
    if 'input_shape' not in config:
        config['input_shape']          = (91, 862, 862)
    if 'force_recomputation' not in config:
        config['force_recomputation']  = False
    if 'gpu_list' not in config:
        config['gpu_list']             = list(range(8))
    if 'postprocess' not in config:
        config['postprocess'] ='clip_float32_to_uint8'

    # by default conditioned on above
    if 'raw_path' not in config:
        config['raw_path']             = os.path.join('/groups/saalfeld/home/heinrichl/data/cremi-2017/',
                                                      'sample_{0:}_cleftsorig_withvalidation.n5'.format(config['sample']))
    if 'meta_path' not in config:
        config['meta_path']            = '/nrs/saalfeld/heinrichl/cremi_meta/{0:}/'.format(config['sample'])
    if 'output_path' not in config:
        config['output_path']          = os.path.join(config['output_base'], config['experiment_name'])
    if 'out_file' not in config:
        config['out_file']             = os.path.join(config['output_path'], config['sample']+'.n5')
    if 'target_keys' not in config:
        config['target_keys']           = ('it_{0:}'.format(config['iteration']),)
    if 'blocklist_file' not in config:
        config['blocklist_file']       = '{0:}_z{1:}_y{2:}_x{3:}.json'.format('validation',
                                                                              config['output_shape'][0],
                                                                              config['output_shape'][1],
                                                                              config['output_shape'][2])
    if 'weight_meta_graph' not in config:
        config['weight_meta_graph']    = os.path.join(config['output_path'],'unet_checkpoint_{0:}'.format(
            config['iteration']))
    if 'inference_meta_graph' not in config:
        config['inference_meta_graph'] = os.path.join(config['output_path'], 'unet_inference')
    if 'net_io_json' not in config:
        config['net_io_json']          = os.path.join(config['output_path'], 'net_io_names.json')
    if 'offset_list_name_extension' not in config:
        config['offset_list_name_extension'] = '_{0:}_z{1:}_y{2:}_x{3:}'.format('validation',
                                                                                config['output_shape'][0],
                                                                                config['output_shape'][1],
                                                                                config['output_shape'][2])

    with open(config['net_io_json'], 'r') as f:
        net_io_names = json.load(f)
    if 'input_key' not in config:
        config['input_key'] = net_io_names[config['net_io_input']]
    if 'output_key' not in config:
        config['output_key'] = net_io_names[config['net_io_output']]
    return config


if __name__=='__main__':
    #experiment_names = ['DTU2_Bonly']
    #experiment_names = [
    #    'baseline_DTU2',
    #    'DTU2_unbalanced',
    #    'DTU2-small',
    #    'DTU2_100tanh',
    #    'DTU2_150tanh',
    #    'DTU2_Aonly',
    #    #'DTU2_Bonly',
    #    'DTU2_Conly',
    #    'DTU2_Adouble',
    #]

    #experiment_names = [
    #    'baseline_DTU1',
    #    'DTU1_unbalanced'
    #]
    #input_shape = (88, 808, 808)
    #output_shape = (60, 596, 596)

    #experiment_names=['DTU2_plus_bdy']
    #net_io_output='syn_dist'

    #experiment_names = ['DTU1_plus_bdy']
    #net_io_output='syn_dist'
    #input_shape = (88, 808, 808)
    #output_shape = (60, 596, 596)

    #experiment_names = ['BCU1']
    #net_io_output='probabilities'  # not necessary, overwritten with output_key
    #output_key = 'Reshape_3:0'
    #input_shape = (88, 808, 808)
    #output_shape = (60, 596, 596)
    #postprocess = 'clip_float32_to_uint8_range_0_1'

    experiment_names = ['BCU2']
    net_io_output = 'probabilities'  # not necessary, overwritten with output_key
    output_key = 'Reshape_3:0'
    postprocess = 'clip_float32_to_uint8_range_0_1'



    for experiment_name in experiment_names:
        for sample in ['A', 'B', 'C']:
            for iteration in range(70000,84000,2000):
                config = configure(experiment_name=experiment_name, sample=sample,
                                   iteration=iteration,
                                   net_io_output=net_io_output,
                                   output_key=output_key,
                                   #input_shape=input_shape, output_shape=output_shape,
                                   postprocess=postprocess
                               )
                with open('config_'+config['experiment_name']+'_'+config['sample']+'_'+str(config['iteration'])+'.json', 'w') as f:
                    json.dump(config, f)