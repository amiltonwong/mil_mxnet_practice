import mxnet as mx
from collections import namedtuple

Conv_Unit = namedtuple('Conv_Unit', ['identity_mapping', 'kernels', 'filters', 'strides', 'dilates', 'dropout', 'lr_mult', 'wd_mult', 'use_global_stats', 'fix_gamma', 'has_bias', 'layers'])

Conv_Stage = namedtuple('Conv_Stage', ['stage_name','num_of_unit','list_of_units'])

def resnet34(lr_mult, wd_mult):
    conv_stages = [                      #identity       kernel      filters            stride       dilate      dropout    lr_mult  wd_mult      use_global fix_gamma has_bias
        Conv_Stage('stage1', 3, [Conv_Unit(False,        [3,3,1],    [64,64,64],        [1,1,1],     [1,1,1],    0.,        lr_mult, wd_mult,     True,      False,    False,         3)]+
                                [Conv_Unit(True,         [3,3],      [64,64],           [1,1],       [1,1],      0.,        lr_mult, wd_mult,     True,      False,    False,         2)]*2),

        Conv_Stage('stage2', 4, [Conv_Unit(False,        [3,3,1],    [128,128,128],     [2,1,2],     [1,1,1],    0.,        lr_mult, wd_mult,     True,      False,    False,         3)]+
                                [Conv_Unit(True,         [3,3],      [128,128],         [1,1],       [1,1],      0.,        lr_mult, wd_mult,     True,      False,    False,         2)]*3),

        Conv_Stage('stage3', 6, [Conv_Unit(False,        [3,3,1],    [256,256,256],     [1,1,1],     [1,2,1],    0.,        lr_mult, wd_mult,     True,      False,    False,         3)]+
                                [Conv_Unit(True,         [3,3],      [256,256],         [1,1],       [2,2],      0.,        lr_mult, wd_mult,     True,      False,    False,         2)]*5),

        Conv_Stage('stage4', 3, [Conv_Unit(False,        [3,3,1],    [512,512,512],     [1,1,1],     [2,4,2],    0.,        lr_mult,  wd_mult,    True,      False,    False,         3)]+
                                [Conv_Unit(True,         [3,3],      [512,512],         [1,1],       [4,4],      0.,        lr_mult,  wd_mult,    True,      False,    False,         2)]*2),
    ]
    return conv_stages

def resnet101(lr_mult, wd_mult):
    conv_stages = [                       #identity    kernel         filters              stride         dilate        dropout    lr_mult  wd_mult      use_global  fix_gamma  has_bias
        Conv_Stage('stage1', 3, [Conv_Unit(False,      [1,3,1,1],     [64,64,256,256],     [1,1,1,1],     [1,1,1,1],    0.,        lr_mult, wd_mult,     True,       False,     False,     4)]+
                                [Conv_Unit(True,       [1,3,1],       [64,64,256],         [1,1,1],       [1,1,1],      0.,        lr_mult, wd_mult,     True,       False,     False,     3)]*2),

        Conv_Stage('stage2', 4, [Conv_Unit(False,      [1,3,1,1],     [128,128,512,512],   [1,2,1,2],     [1,1,1,1],    0.,        lr_mult, wd_mult,     True,       False,     False,     4)]+
                                [Conv_Unit(True,       [1,3,1],       [128,128,512],       [1,1,1],       [1,1,1],      0.,        lr_mult, wd_mult,     True,       False,     False,     3)]*3),

        Conv_Stage('stage3', 23, [Conv_Unit(False,     [1,3,1,1],     [256,256,1024,1024], [1,1,1,1],     [1,2,1,1],    0.,        lr_mult, wd_mult,     True,       False,     False,     4)]+
                                 [Conv_Unit(True,      [1,3,1],       [256,256,1024],      [1,1,1],       [1,2,1],      0.,        lr_mult, wd_mult,     True,       False,     False,     3)]*22),

        Conv_Stage('stage4', 3, [Conv_Unit(False,      [1,3,1,1],     [512,512,2048,2048], [1,1,1,1],     [1,4,1,1],    0.,        lr_mult,  wd_mult,    True,       False,     False,     4)]+
                                [Conv_Unit(True,       [1,3,1],       [512,512,2048],      [1,1,1],       [1,4,1],      0.,        lr_mult,  wd_mult,    True,       False,     False,     3)]*2),
    ]
    return conv_stages

class resnet_wrap(object):
    def __init__(self, net_name, input, lr_mult, wd_mult, workspace):
        self._conv_stages = self.get_resnet_stages(resnet_name=net_name, lr_mult=lr_mult, wd_mult=wd_mult)
        self._endpoints_dict = {}
        self._lr_mult = lr_mult
        self._wd_mult = wd_mult
        self._workspace = workspace

        data = input
        conv0_info = Conv_Unit(False, [7], [64], [2], [1], 0., self._lr_mult, self._wd_mult, True, False, False, 1)
        data = self._bn_(data=data, name='bn_data', bn_id=-1, bn_info=conv0_info)
        self._endpoints_dict['bn_data'] = data
        data = self._conv_(data=data, name='conv0', conv_id=-1,
                      conv_info=conv0_info, workspace=self._workspace, group=1)
        data = self._bn_(data=data, name='bn0', bn_id=-1, bn_info=conv0_info)
        data = mx.sym.Activation(data=data,
                                  name='relu0',
                                  act_type='relu')
        self._endpoints_dict['relu0'] = data
        data = mx.sym.Pooling(data=data,
                              name='pooling0',
                              global_pool=False,
                              kernel=(3,3),
                              pad=(1,1),
                              pool_type='max',
                              stride=(2,2))

        for current_stage in self._conv_stages:
            stage_name = current_stage.stage_name
            cur_stage_units = current_stage.list_of_units
            cur_num_units = current_stage.num_of_unit
            for uid in range(cur_num_units):
                unit_name = 'unit{}'.format(uid+1)
                data = self.unit_fit(stage_name=stage_name, unit_name=unit_name, data=data, unit_info=cur_stage_units[uid], lr_mult=lr_mult, wd_mult=wd_mult)

        bn1 = self._bn_(data=data, name='bn1', bn_id=-1, bn_info=conv0_info)
        relu1 = mx.sym.Activation(data=bn1,
                                  name='relu1',
                                  act_type='relu')
        self._endpoints_dict['relu1'] = relu1

    def get_resnet_stages(self, resnet_name, lr_mult, wd_mult):
        assert resnet_name in ['resnet101', 'resnet34'], 'unknown network.'
        if resnet_name == 'resnet101':
            return resnet101(lr_mult=lr_mult, wd_mult=wd_mult)
        elif resnet_name == 'resnet34':
            return resnet34(lr_mult=lr_mult, wd_mult=wd_mult)

    def unit_fit(self, stage_name, unit_name, data, unit_info, lr_mult, wd_mult):
        #dropout can be added here
        current_name = '{}_{}'.format(stage_name, unit_name)
        branch2 = data
        branch1 = data

        print 'Stage:',stage_name, unit_info
        if unit_info.identity_mapping:
            for id in range(unit_info.layers):
                branch1 = self._bn_(data=branch1, name=current_name, bn_id=id, bn_info=unit_info)
                branch1 = mx.sym.Activation(data=branch1,
                                  name='{}_relu{}'.format(current_name,id),
                                  act_type='relu')
                self._endpoints_dict['{}_relu{}'.format(current_name,id)] = branch1

                if id == unit_info.layers-1 and unit_info.dropout > 0.:
                    print '#########################Dropout Ratio: {}#########################'.format(unit_info.dropout)
                    branch1 = mx.sym.Dropout(branch1, name='{}_dropout'.format(current_name),
                                   p=unit_info.dropout)
                branch1 = self._conv_(data=branch1, name=current_name, conv_id=id,
                                 conv_info=unit_info, workspace=self._workspace, group=1)

        else:
            for id in range(unit_info.layers-1):
                branch1 = self._bn_(data=branch1, name=current_name, bn_id=id, bn_info=unit_info)
                branch1 = mx.sym.Activation(data=branch1,
                                      name='{}_relu{}'.format(current_name,id),
                                      act_type='relu')
                self._endpoints_dict['{}_relu{}'.format(current_name,id)] = branch1
                if id == 0:
                    branch2 = self._conv_(data=branch1, name=current_name, conv_id=-1,
                                     conv_info=unit_info, workspace=self._workspace, group=1)
                if id == unit_info.layers-2 and unit_info.dropout > 0.:
                    print '##Dropout {}##'.format(unit_info.dropout)
                    branch1 = mx.sym.Dropout(branch1, name='{}_dropout'.format(current_name),
                                   p=unit_info.dropout)
                branch1 = self._conv_(data=branch1, name=current_name, conv_id=id,
                                 conv_info=unit_info, workspace=self._workspace, group=1)
        return branch1+branch2

    def get_endpoint(self, endpoint_name):
        assert endpoint_name in self._endpoints_dict.keys(), 'unknown endpoint.'
        return self._endpoints_dict[endpoint_name]

    def _conv_(self, data, name, conv_id, conv_info, workspace, group):
        if conv_id == -1:
            conv_name = name
            conv_id = conv_info.layers-1
            if conv_info.layers > 1:
                conv_name = '{}_sc'.format(name)
        else:
            conv_name = '{}_conv{}'.format(name, conv_id+1)

        lr_mult = conv_info.lr_mult
        wd_mult = conv_info.wd_mult
        print 'Name: {}_weight  lr_mult: {}  wd_mult: {}'.format(conv_name, lr_mult, wd_mult)

        weight = mx.sym.Variable(name='{}_weight'.format(conv_name),
                                 lr_mult=lr_mult,
                                 wd_mult=wd_mult,
                                 init=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=0.005),
                                 )

        kernel = conv_info.kernels[conv_id]
        num_filter = conv_info.filters[conv_id]
        stride = conv_info.strides[conv_id]
        dilate = conv_info.dilates[conv_id]
        pad = ( (kernel-1)*dilate+1 )//2

        if conv_info.has_bias:
            bias = mx.sym.Variable(name='{}_bias'.format(conv_name),
                                   lr_mult=2.0*lr_mult,
                                   wd_mult=0.0*wd_mult,
                                   init=mx.init.Zero(),
                                   )
            print 'Name: {}_bias  lr_mult: {}  wd_mult: {}'.format(conv_name, lr_mult*2.0, wd_mult*0.0)

            return mx.sym.Convolution(data=data,
                                      weight=weight,
                                      bias=bias,
                                      name=conv_name,
                                      kernel=(kernel, kernel),
                                      stride=(stride, stride),
                                      dilate=(dilate, dilate),
                                      pad=(pad, pad),
                                      num_filter=num_filter,
                                      num_group=group,
                                      workspace=workspace,
                                      no_bias=False)
        else:
            return mx.sym.Convolution(data=data,
                                      weight=weight,
                                      name=conv_name,
                                      kernel=(kernel, kernel),
                                      stride=(stride, stride),
                                      dilate=(dilate, dilate),
                                      pad=(pad, pad),
                                      num_filter=num_filter,
                                      num_group=group,
                                      workspace=workspace,
                                      no_bias=True)

    def _bn_(self, data, name, bn_id, bn_info, eps=1.001e-5):
        if bn_id == -1:
            bn_name = name
        else:
            bn_name = '{}_bn{}'.format(name, bn_id+1)

        lr_mult = bn_info.lr_mult
        wd_mult = bn_info.wd_mult
        fix_gamma = bn_info.fix_gamma
        use_global_stats = bn_info.use_global_stats

        print 'Name: {}_gamma  lr_mult: {}  wd_mult: {}'.format(bn_name, lr_mult, wd_mult)
        print 'Name: {}_beta   lr_mult: {}  wd_mult: {}'.format(bn_name, lr_mult*2.0, wd_mult*0.0)

        gamma = mx.sym.Variable('{}_gamma'.format(bn_name),
                                lr_mult=lr_mult,
                                wd_mult=wd_mult,
                                )
        beta = mx.sym.Variable('{}_beta'.format(bn_name),
                               lr_mult=2.0*lr_mult,
                               wd_mult=0.0*wd_mult,
                               )
        return mx.sym.BatchNorm(data = data,
                                gamma = gamma,
                                beta = beta,
                                name = bn_name,
                                eps = eps,
                                fix_gamma = fix_gamma,
                                use_global_stats = use_global_stats)

def fcn34():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    basenetwork = resnet_wrap(net_name='resnet34', input=data, lr_mult=1.0, wd_mult=1.0, workspace=1024)
    last_relu = basenetwork.get_endpoint('relu1')

    conv_params_dict = {'c_kernel' : 3,
                        'c_filters' : 512,
                        'c_stride' : 1,
                        'c_dilate' : 12,
                        'c_lr_mult' : 1.0,
                        'c_wd_mult' : 1.0,
                        'padding' : True,
                        'no_bias' : False,
                        'has_bn' : False,
                        'has_act' : False,
                        'workspace' : 1024,
                        }
    dim_reduc = ConvUnit(data=last_relu, name='dim_reduc', params_dict=conv_params_dict)
    cls = Classifier(raw_feat=dim_reduc, label=label, num_classes=21,prefix='cls')
    return cls

def ConvUnit(data, name, params_dict):
    c_kernel = params_dict['c_kernel']
    c_num_filter = params_dict['c_filters']
    c_stride = params_dict['c_stride']
    c_dilate = params_dict['c_dilate']
    c_lr_mult = params_dict['c_lr_mult']
    c_wd_mult = params_dict['c_wd_mult']
    workspace = params_dict['workspace']

    print 'Name: {}_weight  lr_mult: {}  wd_mult: {} stride: {} dilate: {}'.format(name, c_lr_mult, c_wd_mult, c_stride, c_dilate)

    if params_dict['padding']:
        c_pad = ( (c_kernel-1)*c_dilate+1 )//2
    else:
        c_pad = 0
    c_weight = mx.sym.Variable(name='{}_conv_weight'.format(name),
                               lr_mult=c_lr_mult,
                               wd_mult=c_wd_mult,
                               init=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=0.05),
                              )
    if params_dict['no_bias']:
        net = mx.sym.Convolution(data=data,
                                 weight=c_weight,
                                 name='{}_conv'.format(name),
                                 kernel=(c_kernel, c_kernel),
                                 stride=(c_stride, c_stride),
                                 dilate=(c_dilate, c_dilate),
                                 pad=(c_pad, c_pad),
                                 num_filter=c_num_filter,
                                 num_group=1,
                                 workspace=workspace,
                                 no_bias=params_dict['no_bias'])
    else:
        bias = mx.sym.Variable(name='{}_conv_bias'.format(name),
                               lr_mult=c_lr_mult*2,
                               wd_mult=0,
                               init=mx.init.Zero()
                               )
        print 'Name: {}_conv_bias  lr_mult: {}  wd_mult: {}'.format(name, c_lr_mult*2, 0)
        net = mx.sym.Convolution(data=data,
                                 weight=c_weight,
                                 bias=bias,
                                 name='{}_conv'.format(name),
                                 kernel=(c_kernel, c_kernel),
                                 stride=(c_stride, c_stride),
                                 dilate=(c_dilate, c_dilate),
                                 pad=(c_pad, c_pad),
                                 num_filter=c_num_filter,
                                 num_group=1,
                                 workspace=workspace,
                                 no_bias=params_dict['no_bias'])
    if params_dict['has_bn']:
        gamma = mx.sym.Variable('{}_c_bn_gamma'.format(name),
                            lr_mult=c_lr_mult,
                            wd_mult=c_wd_mult,
                            )

        print 'Name: {}_bn_gamma  lr_mult: {}  wd_mult: {}'.format(name, c_lr_mult, c_wd_mult)

        beta = mx.sym.Variable('{}_c_bn_beta'.format(name),
                           lr_mult=c_lr_mult*2,
                           wd_mult=0.0,
                           )

        print 'Name: {}_bn_beta  lr_mult: {}  wd_mult: {}'.format(name, 2*c_lr_mult, 0)

        net = mx.sym.BatchNorm(data=net,
                               gamma=gamma,
                               beta=beta,
                               name='{}_c_bn'.format(name),
                               eps = 1.001e-5,
                               fix_gamma = params_dict['bn_fix_gamma'],
                               use_global_stats = params_dict['bn_use_global_state'])
    if params_dict['has_act']:
        act_type = params_dict['act_type']
        net = mx.sym.Activation(data=net,
                                name='{}_c_act'.format(name),
                                act_type=act_type)

    return net

def Classifier(raw_feat, label, num_classes, grad_scale=1.0, workspace=1024, prefix='dr'):
    conv_params_dict = {'c_kernel' : 3,
                        'c_filters' : num_classes,
                        'c_stride' : 1,
                        'c_dilate' : 12,
                        'c_lr_mult' : 1.0,
                        'c_wd_mult' : 1.0,
                        'padding' : True,
                        'no_bias' : False,
                        'has_bn' : False,
                        'has_act' : False,
                        'workspace' : workspace,
                        }
    net = ConvUnit(data=raw_feat, name='{}_{}'.format(prefix, num_classes), params_dict=conv_params_dict)
    return mx.sym.SoftmaxOutput(data=net,
                                label=label,
                                ignore_label = 255,
                                use_ignore = True,
                                name='softmax',
                                grad_scale=grad_scale,
                                multi_output=True,
                                normalization='valid'
                                )

sym = mx.sym.load('resnet-34-symbol.json')

mx.viz.plot_network(sym, shape={'data': (1, 3, 320, 320)}, node_attrs={'fixedsize':'false'}).view('res34')

#mx.viz.plot_network(fcn34(), shape={'data':  (1, 3, 320, 320), 'label': (1, 1, 320/8, 320/8)}, node_attrs={'fixedsize': 'false'}).view('fcn34')