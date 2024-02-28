# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023.HF4 replay file
# Internal Version: 2023_07_21-20.45.57 RELr425 183702
# Run by pyoy1 on Sat Feb 10 17:54:19 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=213.143753051758, 
    height=163.390747070312)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
import os
os.chdir(os.getcwd())

executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
o1 = session.openOdb(name='nano_umat.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Warning: The output database file has been loaded, but its format does not match the latest format supported by this version of ABAQUS.
#: 
#: Some functionality may not be supported, and application instability may result.
#: 
#:  It is strongly recommended that you close the database and re-open it using the correct ABAQUS version.
#: Model: Z:/Documents/young/RA/nano_umat.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     2
#: Number of Meshes:             2
#: Number of Element Sets:       11
#: Number of Node Sets:          14
#: Number of Steps:              1

odb = session.odbs['nano_umat.odb']
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF3'), )), ('U', NODAL, ((COMPONENT, 'U3'), )), ), 
    nodeSets=("REFERENCE_POINT_ASSEMBLY_SET-INDENTER-1    67396", ))
xy1 = session.xyDataObjects['RF:RF3 PI: ASSEMBLY_SET-INDENTER-1 N: 67396']
xy2 = -xy1
xy2.setValues(
    sourceDescription=' - "RF:RF3 PI: ASSEMBLY_SET-INDENTER-1 N: 67396"')
tmpName = xy2.name
session.xyDataObjects.changeKey(tmpName, 'Pressure')
xy1 = session.xyDataObjects['U:U3 PI: ASSEMBLY_SET-INDENTER-1 N: 67396']
xy2 = -xy1/1000
xy2.setValues(
    sourceDescription=' - "U:U3 PI: ASSEMBLY_SET-INDENTER-1 N: 67396"/1000')
tmpName = xy2.name
session.xyDataObjects.changeKey(tmpName, 'Height')
xy1 = session.xyDataObjects['Height']
xy2 = session.xyDataObjects['Pressure']
xy3 = combine(xy1, xy2)
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
c1 = session.Curve(xyData=xy3)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
xy1 = session.xyDataObjects['Height']
xy2 = session.xyDataObjects['Pressure']
xy3 = combine(xy1, xy2)
xy3.setValues(sourceDescription='combine ( "Height","Pressure" )')
tmpName = xy3.name
session.xyDataObjects.changeKey(tmpName, 'ph_curve')

x0 = session.xyDataObjects['Height']
x1 = session.xyDataObjects['Pressure']

session.writeXYReport(fileName='PH_Curve.txt', xyData=(x0, x1))
