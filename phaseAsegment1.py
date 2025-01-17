# generating population-level parameters and connection tables
# important output variables
# - popdata
# - pathways
# - Connectivity_AMPA
# - MeanEff_AMPA
# - Connectivity_GABA
# - MeanEff_GABA
# - Connectivity_NMDA
# - MeanEff_NMDA


from frontendhelpers import *


celldefaults = ParamSet('celldef', {
    'N': 75,
    'C': 0.5,
    'Taum': 20,
    'RestPot': -70,
    'ResetPot': -55,
    'Threshold': -50,

    'RestPot_ca': -85,
    'Alpha_ca': 0.5,
    'Tau_ca': 80,
    'Eff_ca': 0.0,

    'tauhm': 20,
    'tauhp': 100,
    'V_h': -60,
    'V_T': 120,
    'g_T': 0,

    'g_adr_max': 0,
    'Vadr_h': -100,
    'Vadr_s': 10,
    'ADRRevPot': -90,
    'g_k_max': 0,
    'Vk_h': -34,
    'Vk_s': 6.5,
    'tau_k_max': 8,
    'n_k': 0,
    'h': 1,

})


popspecific = {
    'LIP': {'N': 204},
    'FSI': {'C': 0.2, 'Taum': 10},
    # should be 10 but was 20 due to bug
    'GPeP': {'N': 750, 'g_T': 0.06, 'Taum': 20},
    'STNE': {'N': 750, 'g_T': 0.06},
    'LIPI': {'N': 186, 'C': 0.2, 'Taum': 10},
    'Th': {'Taum': 27.78}
}


receptordefaults = ParamSet('receptordef', {
    'Tau_AMPA': 2,
    'RevPot_AMPA': 0,

    'Tau_GABA': 5,
    'RevPot_GABA': -70,

    'Tau_NMDA': 100,
    'RevPot_NMDA': 0,
})


basestim = {
    'FSI': {
        'FreqExt_AMPA': 3.6,
        'MeanExtEff_AMPA': 1.55,
        'MeanExtCon_AMPA': 800},
    'LIPI': {
        'FreqExt_AMPA': 1.05,
        'MeanExtEff_AMPA': 1.2,
        'MeanExtCon_AMPA': 640},
    'GPi': {
        'FreqExt_AMPA': 0.8,
        'MeanExtEff_AMPA': 5.9,
        'MeanExtCon_AMPA': 800},
    'STNE': {
        'FreqExt_AMPA': 4.45,
        'MeanExtEff_AMPA': 1.65,
        'MeanExtCon_AMPA': 800},
    'GPeP': {
        'FreqExt_AMPA': 4,
        'MeanExtEff_AMPA': 2,
        'MeanExtCon_AMPA': 800,
        'FreqExt_GABA': 2,
        'MeanExtEff_GABA': 2,
        'MeanExtCon_GABA': 2000},
    'D1STR': {
        'FreqExt_AMPA': 1.3,
        'MeanExtEff_AMPA': 4,
        'MeanExtCon_AMPA': 800},
    'D2STR': {
        'FreqExt_AMPA': 1.3,
        'MeanExtEff_AMPA': 4,
        'MeanExtCon_AMPA': 800},
    'LIP': {
        'FreqExt_AMPA': 2.2,
        'MeanExtEff_AMPA': 2,
        'MeanExtCon_AMPA': 800},
    'Th': {
        'FreqExt_AMPA': 2.2,
        'MeanExtEff_AMPA': 2.5,
        'MeanExtCon_AMPA': 800},
}


dpmndefaults = ParamSet('dpmndef', {
    'dpmn_tauDOP': 2,
    'dpmn_alpha': 0.3,
    'dpmn_DAt': 0.0,
    'dpmn_taum': 1e100,
    'dpmn_dPRE': 0.8,
    'dpmn_dPOST': 0.04,
    'dpmn_tauE': 15,
    'dpmn_tauPRE': 15,
    'dpmn_tauPOST': 6,
    'dpmn_wmax': 0.3,
    'dpmn_w': 0.1286,
    'dpmn_Q1': 0.0,
    'dpmn_Q2': 0.0,
    'dpmn_m': 1.0,
    'dpmn_E': 0.0,
    'dpmn_DAp': 0.0,
    'dpmn_APRE': 0.0,
    'dpmn_APOST': 0.0,
    'dpmn_XPRE': 0.0,
    'dpmn_XPOST': 0.0
})


d1defaults = ParamSet('d1def', {
    'dpmn_type': 1,
    'dpmn_alphaw': 55 / 3.0,  # ???
    'dpmn_a': 1.0,
    'dpmn_b': 0.1,
    'dpmn_c': 0.05,
})


d2defaults = ParamSet('d2def', {
    'dpmn_type': 2,
    'dpmn_alphaw': -45 / 3.0,
    'dpmn_a': 0.5,
    'dpmn_b': 0.005,
    'dpmn_c': 0.05,
})


channels = ParamSet('actionchannel', {
    'action': [1, 2],
})


popdata = pd.DataFrame()
popdata['name'] = [
    'GPi',
    'STNE',
    'GPeP',
    'D1STR',
    'D2STR',
    'LIP',
    'Th',
    'FSI',
    'LIPI',
]
popdata = trace(popdata, 'init')


popdata = ModifyViaSelector(popdata, channels, SelName(
    ['GPi', 'STNE', 'GPeP', 'D1STR', 'D2STR', 'LIP', 'Th']))


popdata = ModifyViaSelector(popdata, celldefaults)


for key, data in popspecific.items():
    params = ParamSet('popspecific', data)
    popdata = ModifyViaSelector(popdata, params, SelName(key))


popdata = ModifyViaSelector(popdata, receptordefaults)


for key, data in basestim.items():
    params = ParamSet('basestim', data)
    popdata = ModifyViaSelector(popdata, params, SelName(key))


popdata = ModifyViaSelector(popdata, dpmndefaults, SelName(['D1STR', 'D2STR']))
popdata = ModifyViaSelector(popdata, d1defaults, SelName('D1STR'))
popdata = ModifyViaSelector(popdata, d2defaults, SelName('D2STR'))


simplepathways = pd.DataFrame(
    [
        ['LIP', 'D1STR', 'AMPA', 'syn', 1, 0.027],
        ['LIP', 'D1STR', 'NMDA', 'syn', 1, 0.027],
        ['LIP', 'D2STR', 'AMPA', 'syn', 1, 0.027],
        ['LIP', 'D2STR', 'NMDA', 'syn', 1, 0.027],
        ['LIP', 'FSI', 'AMPA', 'all', 1, 0.198],
        ['LIP', 'Th', 'AMPA', 'all', 1, 0.035],
        ['LIP', 'Th', 'NMDA', 'all', 1, 0.035],

        ['D1STR', 'D1STR', 'GABA', 'syn', 0.45, 0.28],
        ['D1STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28],
        ['D1STR', 'GPi', 'GABA', 'syn', 1, 2.09],

        ['D2STR', 'D2STR', 'GABA', 'syn', 0.45, 0.28],
        ['D2STR', 'D1STR', 'GABA', 'syn', 0.5, 0.28],
        ['D2STR', 'GPeP', 'GABA', 'syn', 1, 4.07],

        ['FSI', 'FSI', 'GABA', 'all', 1, 3.25833],
        ['FSI', 'D1STR', 'GABA', 'all', 1, 1.7776],
        ['FSI', 'D2STR', 'GABA', 'all', 1, 1.669867],

        ['GPeP', 'GPeP', 'GABA', 'all', 0.0667, 1.75],
        ['GPeP', 'STNE', 'GABA', 'syn', 0.0667, 0.35],
        ['GPeP', 'GPi', 'GABA', 'syn', 1, 0.06],

        ['STNE', 'GPeP', 'AMPA', 'syn', 0.161668, 0.07],
        ['STNE', 'GPeP', 'NMDA', 'syn', 0.161668, 1.51],
        ['STNE', 'GPi', 'NMDA', 'all', 1, 0.038],

        ['GPi', 'Th', 'GABA', 'syn', 1, 0.3315],

        ['Th', 'D1STR', 'AMPA', 'syn', 1, 0.3825],
        ['Th', 'D2STR', 'AMPA', 'syn', 1, 0.3825],
        ['Th', 'FSI', 'AMPA', 'all', 0.8334, 0.1],
        ['Th', 'LIP', 'NMDA', 'all', 0.8334, 0.03],

        # ramping ctx

        ['LIP', 'LIP', 'AMPA', 'all', 0.4335, 0.0127],
        ['LIP', 'LIP', 'NMDA', 'all', 0.4335, 0.15],
        ['LIP', 'LIPI', 'AMPA', 'all', 0.241667, 0.113],
        ['LIP', 'LIPI', 'NMDA', 'all', 0.241667, 0.525],

        ['LIPI', 'LIP', 'GABA', 'all', 1, 1.75],
        ['LIPI', 'LIPI', 'GABA', 'all', 1, 3.58335],

        ['Th', 'LIPI', 'NMDA', 'all', 0.8334, 0.015],

    ],
    columns=['src', 'dest', 'receptor', 'type', 'con', 'eff']
)
simplepathways = trace(simplepathways, 'init')

#################################3#############################################

pathways = simplepathways.copy()
pathways['biselector'] = None
for idx, row in pathways.iterrows():
    if row['type'] == 'syn':
        pathways.loc[idx, 'biselector'] = NamePathwaySelector(
            row['src'], row['dest'], 'action')
    elif row['type'] == 'all':
        pathways.loc[idx, 'biselector'] = NamePathwaySelector(
            row['src'], row['dest'])
pathways = trace(pathways, 'auto')


connectiongrid = constructSquareDf(untrace(popdata['name'].tolist()))
connectiongrid = trace(connectiongrid, 'init')


Connectivity_AMPA = connectiongrid.copy()
MeanEff_AMPA = connectiongrid.copy()
Connectivity_GABA = connectiongrid.copy()
MeanEff_GABA = connectiongrid.copy()
Connectivity_NMDA = connectiongrid.copy()
MeanEff_NMDA = connectiongrid.copy()


for idx, row in pathways.iterrows():
    biselector = row['biselector']
    receptor = row['receptor']
    con = row['con']
    eff = row['eff']
    if receptor == 'AMPA':
        Connectivity_AMPA = FillGridSelection(
            Connectivity_AMPA, popdata, biselector, con)
        MeanEff_AMPA = FillGridSelection(
            MeanEff_AMPA, popdata, biselector, eff)
    if receptor == 'GABA':
        Connectivity_GABA = FillGridSelection(
            Connectivity_GABA, popdata, biselector, con)
        MeanEff_GABA = FillGridSelection(
            MeanEff_GABA, popdata, biselector, eff)
    if receptor == 'NMDA':
        Connectivity_NMDA = FillGridSelection(
            Connectivity_NMDA, popdata, biselector, con)
        MeanEff_NMDA = FillGridSelection(
            MeanEff_NMDA, popdata, biselector, eff)
