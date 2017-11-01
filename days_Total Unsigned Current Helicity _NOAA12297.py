from __future__ import division, print_function
import matplotlib.pyplot as plt
from matplotlib import dates
import drms

import numpy 
numpy.set_printoptions(threshold=1600)

file = open('testfile.txt','w') 
series = 'hmi.sharp_cea_720s'
sharpnum = 5298 # NOAA12297
kwlist = ['T_REC', 'LON_FWT', 'TOTPOT', 'TOTUSJH', 'TOTUSJZ', 'AREA_ACR']

c = drms.Client()
k = c.query('%s[%d]' % (series, sharpnum), key=kwlist, n='none')
file.write(str(k))
file.close()
k.index = drms.to_datetime(k.T_REC)
t_cm = k.LON_FWT.abs().argmin()
print(k)
plt.rc('axes', titlesize='medium')
plt.rc('axes.formatter', use_mathtext=True)
plt.rc('mathtext', default='regular')
plt.rc('legend', fontsize='medium')

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 6))

axi = ax[0,0]
axi.plot(k.index, k.TOTPOT, '.', ms=2, label='TOTPOT')
axi.set_title('Total Photospheric Magnetic Free Energy')
axi.set_ylabel(r'Ergs $cm^{-1}$',size=15)

axi = ax[0, 1]
axi.plot(k.index, k.TOTUSJH, '.', ms=2, label='TOTUSJH')
axi.set_title('Total Unsigned Current Helicity')
axi.set_ylabel('$G^{2} m^{-1}$',size=15)

axi = ax[1, 0]
axi.plot(k.index, k.AREA_ACR/1e3, '.', ms=2, label='AREA_ACR')
axi.set_title('LoS area of active pixels')
axi.set_ylabel(r'$\mu$Hem $\times 1000$')
axi.set_xlabel('Date')

axi = ax[1, 1]
axi.errorbar(k.index, k.TOTUSJZ, fmt='.', ms=2,
             capsize=0, label='TOTUSJZ')
axi.set_title('Total Unsigned Vertical Current')
axi.set_ylabel(r'Amperes')
axi.set_xlabel('Date')

axi.xaxis.set_major_locator(dates.AutoDateLocator())
axi.xaxis.set_major_formatter(dates.DateFormatter('%b\n%d'))

for axi in ax.flatten():
    axi.axvline(t_cm, ls='--', color='r')
    axi.legend(loc='best', numpoints=1)

fig.tight_layout(pad=1.2, w_pad=2)
#plt.draw()
#plt.show()
#fig.savefig('C:\Users\Pussainsa LAPAN\Pictures\sharp_metadata.pdf')
#fig.savefig('C:\Users\Pussainsa LAPAN\Pictures\sharp_metadata.png', dpi=200)


