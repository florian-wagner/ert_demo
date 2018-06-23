import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
import pybert as pb
from ipywidgets import interact, fixed
from IPython.display import YouTubeVideo

# Geometry definition

def create_geometry(position=0, depth=3, radius=4):
    elecs = np.linspace(-25,25,15) # x coordinate of the electrodes
    # Create geometry definition for the modelling domain.
    # worldMarker=True indicates the default boundary conditions for the ERT
    world = mt.createWorld(start=[-50, 0], end=[50, -50], worldMarker=True)

    # Create a circular heterogeneous body
    block = mt.createCircle(pos=[position, -depth], radius=radius, marker=2,
                            boundaryMarker=0, area=5)

    # Merge geometry definition into a Piecewise Linear Complex (PLC)
    geom = mt.mergePLC([world, block])
    fig, ax = plt.subplots(figsize=(10,6))
    pg.show(geom, ax=ax, hold=True)
    ax.plot(elecs, np.zeros_like(elecs), "kv")
    ax.set_ylim(-20,0)
    return geom

def create_mesh(geom):
    elecs = np.linspace(-25,25,15) # x coordinate of the electrodes

    # Create a Dipole Dipole ('dd') measuring scheme
    scheme = pb.createData(elecs=elecs, schemeName='dd')

    # Put all electrodes (aka. sensors positions) into the geometry to enforce mesh
    # refinement. Due to experience known, its convenient to add further refinement
    # nodes in a distance of 10% of electrode spacing, to achieve sufficient
    # numerical accuracy.
    for pos in scheme.sensorPositions():
        geom.createNodeWithCheck(pos)
        geom.createNodeWithCheck(pos+pg.RVector3(0, -0.5))

    # Create a mesh for the finite element modelling with appropriate mesh quality.
    mesh = mt.createMesh(geom, quality=33)

    # Optional: take a look at the mesh
    fig, ax = plt.subplots(figsize=(10,6))
    pg.show(mesh, ax=ax, hold=True)
    ax.set_ylim(-20,0)
    ax.plot(elecs, np.zeros_like(elecs), "kv")
    pg.show(geom, ax=ax)
    return mesh, scheme

def simulate(res_background, res_anomaly, mesh, scheme):
    # Create a map to set resistivity values in the appropriate regions
    # [[regionNumber, resistivity], [regionNumber, resistivity], [...]
    rhomap = [[1, res_background],
              [2, res_anomaly]]

    # Initialize the ERTManager
    ert = pb.Resistivity()

    # Perform the modeling with the mesh and the measuring scheme itself
    # and return a data container with apparent resistivity values,
    # geometric factors and estimated data errors specified by the noise setting.
    # The noise is also added to the data.
    data = ert.simulate(mesh, res=rhomap, scheme=scheme,
                        noiseLevel=1, noiseAbs=0)

    # Optional: take a look at the data (electrical pseudo section)
    pb.show(data)
    return data

def invert(data):
    ert = pb.Resistivity()
    ert.setData(data)
    ert.createMesh(maxCellArea=3., depth=25)
    model = ert.invert(data, lam=30)

    # Let the ERTManger show you the model and fitting results of the last
    # successful run.
    fig, ax = plt.subplots(figsize=(10,6))
    pg.mplviewer.drawSensors(ax, data.sensorPositions(), color="w")
    ert.showModel(ax=ax)
