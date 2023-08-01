import numpy as np
from matplotlib import pyplot

save_figure = False

# import data sets
PRt, PRr = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=True).T
PR2t, PR2r = np.genfromtxt('gr_q2.txt', delimiter=',', skip_header=True).T
WLt, WLr = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=True).T
Tt, Tte = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=True).T

# plot data and label (production rate and water level)
f, ax1 = pyplot.subplots(nrows=1, ncols=1)
ax2 = ax1.twinx()
ax1.plot(PRt, PRr, 'r-', label='Total Extraction')
ax1.plot(PR2t, PR2r, 'g-', label='Rhyolite Extraction')
ax2.plot(WLt, WLr, 'b-', label='Water level')
ax1.set_ylabel('Extraction Rate (tonnes/day)')
ax2.set_ylabel('Water Level (m)')
ax1.set_xlabel('Time (yr)')
ax1.set_title('Extraction Rate and Water Level versus Time')
ax1.legend(loc=2)
ax2.legend(loc=4)

if not save_figure:
    pyplot.show()
else:
    # save figure to given file name
    pyplot.savefig('production_waterlevel_plot.png', dpi=600)

# plot data and label (production rate and temperature)
f2, ax3 = pyplot.subplots(nrows=1, ncols=1)
ax4 = ax3.twinx()
ax3.plot(PRt, PRr, 'r-', label='Total Extraction')
ax3.plot(PR2t, PR2r, 'g-', label='Rhyolite Extraction')
ax4.plot(Tt, Tte, 'b-', label='Temperature')
ax3.set_ylabel('Extraction Rate (tonnes/day)')
ax4.set_ylabel('Temperature (degC)')
ax3.set_xlabel('Time (yr)')
ax3.set_title('Extraction Rate and Temperature versus Time')
ax3.legend(loc=1)
ax4.legend(loc=2)

if not save_figure:
    pyplot.show()
else:
    # save figure to given file name
    pyplot.savefig('production_temperature_plot.png', dpi=600)