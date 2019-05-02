# __author: Bao Li__ #
import numpy
# Import matplotlib.pyplot for plotting functions
import matplotlib.pyplot as plt


def plotHist(_x_hist=[], _f_hist=[], _c_eq_hist=[], _c_ineq_hist=[]):
    '''
    Function for plotting variable history of scipy.optimize.minimize
    routine.

    Inputs:
        _x_hist, _f_hist, _c_hist : lists holding dv, obj, and con value histories.
    '''
    # Store numpy arrays of dv, obj, and con histories
    x_hist = numpy.array(_x_hist)
    f_hist = _f_hist
    c_eq_hist = numpy.array(_c_eq_hist)
    c_ineq_hist = numpy.array(_c_ineq_hist)

    # Start figure number at zero
    fig_count = 0
    # Allocate an empty list to hold figures
    figure_list = []
    # Close any previously open plots
    plt.close("all")

    # Check x_hist is not empty
    if len(x_hist) > 0:
        # Create a numpy array with iteration number for x_hist
        # (ex. array([0, 1, 2, ..., n_iter]))
        x_iterations = range(len(x_hist[:, 0]))
        # Plot Design Variables Histories
        for dv_num in range(len(x_hist[0])):
            figure_list.append(plt.figure(fig_count))
            # Plot design variable with green line and circle data points
            plt.plot(x_iterations, x_hist[:, dv_num], 'g-o')
            plt.xlabel('Iteration')
            plt.ylabel('x%d' % (dv_num))
            plt.title('Design Variable # %d History' % (dv_num))
            # Increment figure number
            fig_count += 1

    # Check f_hist is not empty
    if len(f_hist) > 0:
        # Create a numpy array with iteration number for f_hist
        f_iterations = range(len(f_hist))
        # Now Plot Objective Function History
        figure_list.append(plt.figure(fig_count))
        # Plot objective with red line and circle data points
        plt.plot(f_iterations, f_hist, 'r-o')
        plt.xlabel('Iteration')
        plt.ylabel('f')
        plt.title('Objective Function History')
        # Increment figure number
        fig_count += 1

        # Check c_eq_hist is not empty
    if len(c_eq_hist) > 0:
        # Create a numpy array with iteration number for c_eq_hist
        c_eq_iterations = range(len(c_eq_hist[:, 0]))
        # Now plot constraint history
        for con_num in range(len(c_eq_hist[0])):
            figure_list.append(plt.figure(fig_count))
            # Plot constraint with blue line and circle data points
            plt.plot(c_eq_iterations, c_eq_hist[:, con_num], 'b-o')
            plt.xlabel('Iteration')
            plt.ylabel('c%d' % (con_num))
            plt.title('Equality Constraint # %d History' % (con_num))
            # Increment figure number
            fig_count += 1

    # Check c_ineq_hist is not empty
    if len(c_ineq_hist) > 0:
        # Create a numpy array with iteration number for c_ineq_hist
        c_iterations = range(len(c_ineq_hist[:, 0]))
        # Now plot constraint history
        for con_num in range(len(c_ineq_hist[0])):
            figure_list.append(plt.figure(fig_count))
            # Plot constraint with blue line and circle data points
            plt.plot(c_iterations, c_ineq_hist[:, con_num], 'b-o')
            plt.xlabel('Iteration')
            plt.ylabel('c%d' % (con_num))
            plt.title('Inequality Constraint # %d History' % (con_num))
            # Increment figure number
            fig_count += 1

    # Show plot!
    plt.show()


def plotContour(obj, obj1, bnds, x_hist=None, con_eq=None, con_ineq=None):
    # Check if dv history list was provided by user, if so make into array
    if x_hist:
        x_hist_final = numpy.array(x_hist)

    # Create a mesh of 2D points to interpolate the objective contours with
    x = numpy.linspace(bnds[0][0], bnds[0][1], 300)
    y = numpy.linspace(bnds[1][0], bnds[1][1], 300)
    X, Y = numpy.meshgrid(x, y)
    F = numpy.zeros_like(X)

    X1, Y1 = numpy.meshgrid(x, y)
    F1 = numpy.zeros_like(X)

    # Check if an equality function was provided
    if con_eq:
        # Get number of equality constraints
        ncon_eq = len(con_eq(numpy.array([X[0, 0], Y[0, 0]])))
        # Create a 3D array to hold all eq constraint values
        # dimensions : (num_con, num_xpts, num_y_pts)
        C_eq = numpy.zeros([ncon_eq, X.shape[0], X.shape[1]])

    # Check if an inequality constraint function was provided
    if con_ineq:
        # Get number of inequality constraints
        ncon_ineq = len(con_ineq(numpy.array([X[0, 0], Y[0, 0]])))
        # Create a 3D array to hold all ineq constraint values
        # dimensions : (num_con, num_xpts, num_y_pts)
        C_ineq = numpy.zeros([ncon_ineq, X.shape[0], X.shape[1]])
    # Dummy values for maximum/minimum objective values, used for contour scaling
    max_obj = -numpy.inf
    min_obj = numpy.inf

    # Loop through each (x,y) pair and evaluate the objective and constraints.
    for i in range(len(x)):
        for j in range(len(y)):
            F[i, j] = obj(numpy.array([X[i, j], Y[i, j]]))
            if F[i, j] > max_obj:
                # If current point is higher then the current maximum value, update.
                max_obj = F[i, j]
            if F[i, j] < min_obj:
                # If current point is lower then the current minimum value, update.
                min_obj = F[i, j]
            if con_eq:
                C_eq[:, i, j] = con_eq(numpy.array([X[i, j], Y[i, j]]))
            if con_ineq:
                C_ineq[:, i, j] = con_ineq(numpy.array([X[i, j], Y[i, j]]))

    # Loop through each (x,y) pair and evaluate the objective and constraints.
    for i in range(len(x)):
        for j in range(len(y)):
            F1[i, j] = obj1(numpy.array([X1[i, j], Y1[i, j]]))
            if F1[i, j] > max_obj:
                # If current point is higher then the current maximum value, update.
                max_obj = F[i, j]
            if F1[i, j] < min_obj:
                # If current point is lower then the current minimum value, update.
                min_obj = F1[i, j]

    # numpy.linspace(min_obj, max_obj), cmap = plt.get_cmap('jet'),

    # Plot obj contours and constraints
    plt.figure(-1)
    cp = plt.contour(X, Y, F, 20, colors='b')
    cp1 = plt.contour(X1, Y1, F1, 20, colors='g')
    plt.clabel(cp, inline=True)
    plt.clabel(cp1, inline=True)
    # Plot optimization path as blue line
    plt.plot(x_hist_final[:, 0], x_hist_final[:, 1], 'k-o')
    # Plot last point as green dot
    start = numpy.array([10.0, 20.0])
    plt.plot(start[0], start[1], 'yo', markersize=6)
    plt.plot(x_hist_final[-1, 0], x_hist_final[-1, 1], 'bo', markersize=6)
    uncon = numpy.array([14.27881924, 11.67599726])
    plt.plot(uncon[0], uncon[1], 'go', markersize=6)

    # Label stuff
    plt.legend(['optimization path', 'starting point', 'constrained minimum', 'unconstrained minimum'])
    plt.xlabel('A(Aspect Ratio)')
    plt.ylabel('S(Wing Area)')
    plt.title('SQP Method')

    # Check if constraints are provided, if so plot them.
    if con_eq:
        for con_num in range(ncon_eq):
            plt.contour(X, Y, C_eq[con_num, :, :], numpy.array([0]), colors='r')
    if con_ineq:
        for con_num in range(ncon_ineq):
            plt.contourf(X, Y, C_ineq[con_num, :, :], numpy.array([0, 1e20]), colors='r', alpha=0.5)

    # Set plot bounds to user-specifed values.
    plt.xlim([bnds[0][0], bnds[0][1]])
    plt.ylim([bnds[1][0], bnds[1][1]])

    # Show plot!
    plt.show()