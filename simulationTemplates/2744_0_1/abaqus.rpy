# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2017 replay file
# Internal Version: 2016_09_27-23.54.59 126836
# Run by nhabibi on Mon Sep 09 15:29:44 2019
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=172.776031494141, 
    height=233.891662597656)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='A200_4500_370_1.3_id1441.odb')
#: Model: O:/Werkstoffmechanik/01 WM MMD/03 Wenqi/01 Project/01 TOOLKIT/03 Nanoindentation/01 DP1000/Simulation/singleCrystal/n=20/200_4500_370_1.3_id1441/A200_4500_370_1.3_id1441.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     2
#: Number of Meshes:             2
#: Number of Element Sets:       11
#: Number of Node Sets:          12
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.0710379, 
    farPlane=0.113048, width=0.00456932, height=0.00207854, 
    viewOffsetX=-0.00266862, viewOffsetY=-0.000631916)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.069407, 
    farPlane=0.113509, width=0.00446442, height=0.00203082, cameraPosition=(
    0.0705987, 0.0457015, 0.0242671), cameraUpVector=(-0.632021, 0.489473, 
    -0.600804), cameraTarget=(0.0115316, -0.014018, -0.0133009), 
    viewOffsetX=-0.00260735, viewOffsetY=-0.000617408)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
leaf = dgo.LeafFromPartInstance(partInstanceName=('PART-1-1', ))
session.viewports['Viewport: 1'].odbDisplay.displayGroup.remove(leaf=leaf)
