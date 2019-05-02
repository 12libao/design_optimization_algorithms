# __author: Bao Li__ #
import numpy as np
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


# Create a dictionary to store options about the mesh
mesh_dict = {'num_y' : 9,
             'num_x' : 2,
             'wing_type' : 'rect',
             'symmetry' : True}

# Generate the aerodynamic mesh based on the previous dictionary
mesh = generate_mesh(mesh_dict)

# Create a dictionary with info and options about the aerodynamic
# lifting surface
surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'twist_cp' : np.ones((2)),
            'mesh' : mesh,

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.0,            # CL of the surface at alpha=0
            'CD0' : 0.015,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.18]),      # thickness over chord ratio (NACA0018)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,  # if true, compute viscous drag
            'with_wave' : False,     # if true, compute wave drag
            }

# Create the OpenMDAO problem
prob = Problem()

# Create an independent variable component that will supply the flow
# conditions to the problem.
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=248.136, units='m/s')
indep_var_comp.add_output('alpha', val=5., units='deg')
indep_var_comp.add_output('Mach_number', val=0.84)
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')

# Add this IndepVarComp to the problem model
prob.model.add_subsystem('prob_vars',
    indep_var_comp,
    promotes=['*'])

# Create and add a group that handles the geometry for the
# aerodynamic lifting surface
geom_group = Geometry(surface=surface)
prob.model.add_subsystem(surface['name'], geom_group)

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_group = AeroPoint(surfaces=[surface])
point_name = 'aero_point_0'
prob.model.add_subsystem(point_name, aero_group,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])

name = surface['name']

# Connect the mesh from the geometry component to the analysis point
prob.model.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')

# Perform the connections with the modified names within the
# 'aero_states' group.
prob.model.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')

prob.model.connect(name + '.t_over_c', point_name + '.' + name + '_perf.' + 't_over_c')

# Import the Scipy Optimizer and set the driver of the problem to use
# it, which defaults to an SLSQP optimization method
from openmdao.api import ScipyOptimizeDriver
prob.driver = ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9

recorder = SqliteRecorder("aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
prob.model.add_constraint(point_name + '.wing_perf.CL', equals=0.5)
prob.model.add_objective(point_name + '.wing_perf.CD', scaler=1e4)

# Set up and run the optimization problem
prob.setup()

#from openmdao.api import view_model
#view_model(prob)
# prob.check_partials(compact_print=True)
# exit()
prob.run_driver()

print(prob['aero_point_0.wing_perf.CD'][0])
print(prob['aero_point_0.wing_perf.CL'][0])
print(prob['aero_point_0.CM'][1])
print(prob['wing.twist_cp'])