import crocoddyl as croc
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat

import numpy as np
import meshcat
from os.path import join
import time
import matplotlib.pyplot as plt

path_to_urdf = "h1_2_description/"
urdf_name = "h1_2_handless.urdf"
weights = {
    "x_reg": 1e-3,  # State regularization weight
    "u_reg": 1e-6,  # Control regularization weight
    "com_track": 1e0,
    "x_reg_term": 1e-2,  # Terminal state regularization weight
}

# Load the robot
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    filename = join(path_to_urdf, urdf_name),
    package_dirs = path_to_urdf,
    root_joint = pin.JointModelFreeFlyer(),
)
model.armature = np.ones(model.nv) * 1e-3
data = model.createData()

nq, nv = model.nq, model.nv
nx = nq + nv
ndx = nv * 2
right_foot_name = "right_ankle_roll_link"
left_foot_name = "left_ankle_roll_link"
right_foot_id = model.getFrameId(right_foot_name)
left_foot_id = model.getFrameId(left_foot_name)
q0 = pin.neutral(model)
q0 = np.array([ 0.        ,  0.        ,  0.9832    ,  0.        ,  0.        ,
        0.        ,  1.        ,  0.        , -0.52359878,  0.        ,
        1.04719755, -0.52359878,  0.        ,  0.        , -0.52359878,
        0.        ,  1.04719755, -0.52359878,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ])
v0 = np.zeros(nv)

# Setup the visualization
#print("Setting up the visualizer...")
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

print("Visualizing the initial robot configuration...")
# Get the first foot positions as references for the Baumegarte contact placement corrector
pin.framesForwardKinematics(model, data, q0)
oMleft_foot = data.oMf[left_foot_id]
oMright_foot = data.oMf[right_foot_id]
mean_height = (oMleft_foot.translation[2] + oMright_foot.translation[2]) / 2.0
# Update the initial robot position to have the feet on the ground (this is just for better visualization)
q0[2] -= mean_height
oMleft_foot.translation[2] -= mean_height
oMright_foot.translation[2] -= mean_height
viz.display(q0)

x0 = np.concatenate([q0, v0])  # Initial state
N = 100
dt = 0.02
# Define the reference trajectory to follow
pin.centerOfMass(model, data, q0, compute_subtree_coms=False)
highest = data.com[0][2]
lowest = data.com[0][2] * 0.6
mean_com = (highest + lowest) / 2.0
amplitude = (highest - lowest) / 2.0
traj = [mean_com + amplitude * np.cos(2* np.pi * t / N) for t in range(N)]

# plt.figure()
# plt.plot(traj)
# plt.show()

# Build running models
running_models = []
for t in range(N-1):
    state = croc.StateMultibody(model)
    actuation = croc.ActuationModelFloatingBase(state)
    ## Contacts
    contacts = croc.ContactModelMultiple(state, actuation.nu)
    for cid, placement in zip([right_foot_id, left_foot_id], [oMright_foot, oMleft_foot]):
        contact = croc.ContactModel6D(
            state,
            cid,
            placement,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([200.0, 20.0])
        )
        contacts.addContact(f"{model.frames[cid]}_contact", contact)
    #### Costs ####
    costs = croc.CostModelSum(state, actuation.nu)
    ## State regularization
    xRegResidual = croc.ResidualModelState(state, x0, actuation.nu)
    xRegCost = croc.CostModelResidual(
        state,
        croc.ActivationModelWeightedQuad(np.concatenate((np.zeros(nv), np.ones(nv)))),
        xRegResidual,
    )
    if weights["x_reg"] > 0.0:
        costs.addCost("xReg", xRegCost, weights["x_reg"])
    ## Control regularization
    uRegResidual = croc.ResidualModelControl(state, actuation.nu)
    uRegCost = croc.CostModelResidual(
        state,
        croc.ActivationModelWeightedQuad(np.ones(actuation.nu)),
        uRegResidual,
    )
    if weights["u_reg"] > 0.0:
        costs.addCost("uReg", uRegCost, weights["u_reg"])
    ## Reference trajectory tracking
    comTrackResidual = croc.ResidualModelCoMPosition(state, np.array([0, 0, traj[t]]), actuation.nu)
    comTrackCost = croc.CostModelResidual(
        state,
        croc.ActivationModelWeightedQuad(np.array([0, 0, 1.0])),
        comTrackResidual,
    )
    if weights["com_track"] > 0.0:
        costs.addCost("xRef", comTrackCost, weights["com_track"])
    #### Differential Action Model ####
    dam = croc.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
    )
    am = croc.IntegratedActionModelEuler(
        dam,
        dt
    )
    running_models.append(am)

# Build the terminal model
contacts = croc.ContactModelMultiple(state, actuation.nu)
for cid in [right_foot_id, left_foot_id]:
    contact = croc.ContactModel6D(
        state,
        cid,
        pin.SE3.Identity(),
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        actuation.nu,
        np.array([0.0, 10.0])
    )
    contacts.addContact(f"{model.frames[cid]}_contact", contact)
## Costs
costs = croc.CostModelSum(state, actuation.nu)
## State regularization
xRegResidual = croc.ResidualModelState(state, x0, actuation.nu)
xRegCost = croc.CostModelResidual(
    state,
    croc.ActivationModelWeightedQuad(np.ones(ndx)),
    xRegResidual,
)
if weights["x_reg_term"] > 0.0:
    costs.addCost("xReg", xRegCost, weights["x_reg_term"])
## Action model
dam = croc.DifferentialActionModelContactFwdDynamics(
    state,
    actuation,
    contacts,
    costs,
)
terminal_model = croc.IntegratedActionModelEuler(
    dam,
    dt
)

### Define the shooting problem
problem = croc.ShootingProblem(
    x0=x0,
    runningModels=running_models,
    terminalModel=terminal_model,
)
ddp = croc.SolverFDDP(problem)
ddp.th_stop = 1e-6
ddp.setCallbacks([
    croc.CallbackVerbose(),
])
xs_init = [x0] * N
us_init = [np.zeros(model.nv - 6)] * (N - 1)

print("Starting DDP optimization...")
ddp.solve(xs_init, us_init, 100)

xs_sol = np.array(ddp.xs)
us_sol = np.array(ddp.us)
# Visualize the solution
input("Press Enter to visualize the solution trajectory...")
for i in range(len(xs_sol)):
    viz.display(xs_sol[i, :nq])
    time.sleep(dt)
