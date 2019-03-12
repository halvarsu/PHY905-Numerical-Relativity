import numpy as np
import matplotlib.pyplot as plt


def run_star(star, rho_values = np.linspace(-3.4, -2.2905, 20)):
    """Short script for running several values of central density,
    plotting the value in-integration and gathering the end results."""
    fig,[ax1,ax2,ax3,ax4] = plt.subplots(1,4, figsize = [15,4])
    final_values = []
    integrator = 'dopri5'
    Nr = 200

    b = np.min(rho_values)
    a = np.max(rho_values)
    for logrhoc in rho_values:
        star.set_initial_conditions(rhoc = 10**logrhoc)
        solver,t,y = star.solve_star_ode(integrator = integrator, Nr=Nr)

        # other method, does not work properly
        # solver = star.solve_star_ivp(integrator = integrator)#
        # t, y = solver.t, solver.y.T
        
        # Edge finding ensures that our position is right
        # before pressure < tol
        last_pos = -1 
        final_values.append([logrhoc, t[last_pos]] + list(y[last_pos]))
        
        c = plt.cm.viridis((logrhoc-a) / (b - a))
        ax1.semilogy(t, y[:, 1], label = 'logrhoc {:.2f}'.format(logrhoc),
                color = c)
        ax2.scatter(logrhoc, y[last_pos, 0], c = c)
        ax3.scatter(t[last_pos], y[last_pos, 0], c = c)
        ax4.plot(t, y[:, 0], color = c)

    ax1.set_xlabel('Radius [scaled]')
    ax2.set_xlabel(r'log$_{10}\rho_c$')
    ax3.set_xlabel('Radius [scaled]')
    ax4.set_xlabel('Radius [scaled]')

    ax1.set_ylabel('Pressure [scaled]')
    ax2.set_ylabel('Mass [$M_{\odot}$] [scaled]')
    ax3.set_ylabel('Mass [$M_{\odot}$] [scaled]')
    ax4.set_ylabel('Mass [$M_{\odot}$] [scaled]')
    fig.tight_layout()
    return final_values
