from calibry import Calibration
import pandas as pd
from cytoflow import ImportOp, Tube, PolygonOp, ScatterplotView
from pathlib import Path

# increase matplotlib default figure size and resolution
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 100


data_path = Path('./data').resolve()

### {{{                 --     load files into cytoflow     --
op0 = ImportOp(
    conditions={
        'beads': 'bool',
        'sc': 'bool',
        'ebfpt': 'bool',
        'eyfpt': 'bool',
        'mkatet': 'bool',
        'double': 'category',
    },
    tubes=[
        Tube(
            file=data_path / 'OA_4_double_mkate_006.fcs',
            conditions={
                'beads': False,
                'sc': False,
                'ebfpt': True,
                'eyfpt': True,
                'mkatet': True,
                'double': 'red',
            },
        ),
        Tube(
            file=data_path / 'OA_5_double_ebfp_007.fcs',
            conditions={
                'beads': False,
                'sc': False,
                'ebfpt': True,
                'eyfpt': True,
                'mkatet': True,
                'double': 'blue',
            },
        ),
        Tube(
            file=data_path / 'OA_6_all_colors_008.fcs',
            conditions={
                'beads': False,
                'sc': False,
                'ebfpt': True,
                'eyfpt': True,
                'mkatet': True,
                'double': 'none',
            },
        ),
        Tube(
            file=data_path / 'OA_beads_005.fcs',
            conditions={
                'beads': True,
                'sc': False,
                'ebfpt': True,
                'eyfpt': True,
                'mkatet': True,
                'double': 'none',
            },
        ),
        Tube(
            file=data_path / 'OA_1_ebfp_002.fcs',
            conditions={
                'beads': False,
                'sc': True,
                'ebfpt': True,
                'eyfpt': False,
                'mkatet': False,
                'double': 'none',
            },
        ),
        Tube(
            file=data_path / 'OA_2_eyfp_003.fcs',
            conditions={
                'beads': False,
                'sc': True,
                'ebfpt': False,
                'eyfpt': True,
                'mkatet': False,
                'double': 'none',
            },
        ),
        Tube(
            file=data_path / 'OA_3_mkate_004.fcs',
            conditions={
                'beads': False,
                'sc': True,
                'ebfpt': False,
                'eyfpt': False,
                'mkatet': True,
                'double': 'none',
            },
        ),
        Tube(
            file=data_path / 'OA_unstained_001.fcs',
            conditions={
                'beads': False,
                'sc': False,
                'ebfpt': False,
                'eyfpt': False,
                'mkatet': False,
                'double': 'none',
            },
        ),
    ],
    channels={
        'APC-A': 'APC_A',
        'APC-Alexa 700-A': 'APC_Alexa_700_A',
        'APC-Cy7-A': 'APC_Cy7_A',
        'AmCyan-A': 'AmCyan_A',
        'FITC-A': 'FITC_A',
        'FSC-A': 'FSC_A',
        'FSC-H': 'FSC_H',
        'FSC-W': 'FSC_W',
        'PE-A': 'PE_A',
        'PE-Texas Red-A': 'PE_Texas_Red_A',
        'Pacific Blue-A': 'Pacific_Blue_A',
        'PerCP-Cy5-5-A': 'PerCP_Cy5_5_A',
        'SSC-A': 'SSC_A',
        'SSC-H': 'SSC_H',
        'SSC-W': 'SSC_W',
        'Time': 'Time',
    },
)

xp = op0.apply()

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     gating with cytoflow     --

ogate_1 = PolygonOp(
    name='gate_1',
    xchannel='FSC_A',
    ychannel='SSC_A',
    vertices=[
        (78272.42052772916, 51941.85891190426),
        (36040.13420379757, 12911.555237724035),
        (30610.923214813705, 3417.527829979214),
        (45111.535559396456, 2003.9938281255725),
        (166558.72309258798, 5473.376296575663),
        (230880.5380177539, 17860.19218740414),
        (248821.4000459927, 28306.356342119674),
        (238869.2448058803, 119991.34674587131),
    ],
    xscale='log',
    yscale='log',
)
xp = ogate_1.apply(xp)
ogate_1.default_view(
    subset='(beads == False) and (ebfpt == False) and (eyfpt == False) and (mkatet == False) and (sc == False)'
).plot(xp)


ogate_2 = PolygonOp(
    name='gate_2',
    xchannel='FSC_W',
    ychannel='FSC_A',
    vertices=[
        (72923.63668727828, 118247.41874012665),
        (84103.69710567757, 66915.4973453086),
        (73644.93090782018, 28416.55629919506),
        (55251.92828400201, 13444.745892373128),
        (50924.16296075068, 22356.537801195704),
        (49481.574519666894, 81530.83607577764),
        (55612.57539427295, 113256.81527118602),
    ],
)

xp = ogate_2.apply(xp)

ogate_2.default_view(
    subset='(beads == False) and (ebfpt == False) and (eyfpt == False) and (mkatet == False) and (gate_1 == True) and (gate_2 == True)'
).plot(xp)


ogate_3 = PolygonOp(
    name='gate_3',
    xchannel='SSC_W',
    ychannel='SSC_A',
    vertices=[
        (89805.19896211695, 29725.266356646753),
        (152913.2845760869, 18101.512859984465),
        (119165.64521032758, 6374.622573429896),
        (91155.10453674733, 1410.0885232003507),
        (54707.65402172723, 1439.5336832118796),
        (52007.842872466485, 4724.005580234048),
        (62132.134682194286, 22488.326287480577),
    ],
    yscale='log',
)

xp = ogate_3.apply(xp)
ogate_3.default_view(
    subset='(beads == False) and (ebfpt == False) and (eyfpt == False) and (mkatet == False) and (gate_1 == True) and (gate_2 == True)'
).plot(xp)

##────────────────────────────────────────────────────────────────────────────}}}##

### {{{       --     load into Calibry and initialize calibration     --
ebfp_ctrl = xp.query('ebfpt and sc and gate_1 and gate_2 and gate_3')
eyfp_ctrl = xp.query('eyfpt and sc and gate_1 and gate_2 and gate_3')
mkate_ctrl = xp.query('mkatet and sc and gate_1 and gate_2 and gate_3')
all_ctrl = xp.query(
    """ebfpt and eyfpt and mkatet and not sc and double == "none" \
    and gate_1 and gate_2 and gate_3 and not beads"""
)

empty_ctrl = xp.query(
    """not ebfpt and not eyfpt and not mkatet and not sc and double == "none" \
    and gate_1 and gate_2 and gate_3 and not beads"""
)


controls = {
    "EBFP": ebfp_ctrl.data,
    "EYFP": eyfp_ctrl.data,
    "MKATE": mkate_ctrl.data,
    "All": all_ctrl.data,
    "Empty": empty_ctrl.data,
}
beads_path = data_path / 'OA_beads_005.fcs'

cal = Calibration(
    controls,
    beads_path,
    reference_protein='MKATE',
    use_channels=['PACIFIC_BLUE', 'FITC', 'PE_TEXAS_RED'],
)

##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                 --     fit and plot diagnostics     --
cal.fit()
cal.plot_bleedthrough_diagnostics()
cal.plot_beads_diagnostics()
cal.plot_color_mapping_diagnostics()
##────────────────────────────────────────────────────────────────────────────}}}


ebfp_dg = xp.query(
    """ebfpt and eyfpt and mkatet and not sc and double == "blue" \
    and gate_1 and gate_2 and gate_3 and not beads"""
)

ScatterplotView(
    xchannel='Pacific_Blue_A',
    xscale='log',
    ychannel='PE_Texas_Red_A',
    yscale='log',
    subset=
    '(double == "blue") and (ebfpt == True) and (eyfpt == True) and (mkatet == True) and (gate_1 == True) and (gate_2 == True) and (gate_3 == True)'
).plot(ebfp_dg)


ebfp_calibrated = cal.apply_to_cytoflow_xp(ebfp_dg, include_arbitrary_units=True)

ScatterplotView(
    xchannel='EBFP',
    xscale='log',
    ychannel='MKATE',
    yscale='log',
    subset=
    '(double == "blue") and (ebfpt == True) and (eyfpt == True) and (mkatet == True) and (gate_1 == True) and (gate_2 == True) and (gate_3 == True)'
).plot(ebfp_calibrated,ylim=(1e2, 1e8),xlim=(1e2, 1e8))
