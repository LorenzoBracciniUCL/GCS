
import numpy as np 
import matplotlib.pyplot as plt
import math as math 
import cmath as cmath
from scipy import linalg as linalg
from scipy import integrate as integ
from numba import jit
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import colors
from fractions import Fraction as Fraction
import matplotlib.animation as animation 

def format_latex_numbers(arr):
    formatted = []
    for num in arr:
        frac = Fraction(num).limit_denominator()
        if frac.denominator == 1:
            formatted.append(f'${frac.numerator}$')
        else:
            formatted.append(f'${frac.numerator}/{frac.denominator}$')
    return formatted

def Plot_Phase_Space_First_QRDM_Func(GCS, leged_lables, array_pi, save):

    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    fig, axes = plt.subplots(2, 3,figsize=(6,4), dpi = 350)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)


    max_x = np.max(np.array([np.real(GCS.r_JK_t[:,:,:,0]),np.imag(GCS.r_JK_t[:,:,:,0])]))*1.1
    min_x = np.min(np.array([np.real(GCS.r_JK_t[:,:,:,0]),np.imag(GCS.r_JK_t[:,:,:,0])]))*1.1
    max_p = np.max(np.array([np.real(GCS.r_JK_t[:,:,:,1]),np.imag(GCS.r_JK_t[:,:,:,1])]))*1.1
    min_p = np.min(np.array([np.real(GCS.r_JK_t[:,:,:,1]),np.imag(GCS.r_JK_t[:,:,:,1])]))*1.1


    lables_pi = format_latex_numbers(array_pi)

    for i in range(2):
        for j in range(3):
            axes[i,j].set_xlim(-GCS.t_array[-1]/(20*np.pi), GCS.t_array[-1]*21/(20*np.pi))
            axes[i,j].set_xticks(array_pi)
            axes[i,j].grid()
            axes[i,j].axvline(x=0, color='black',linewidth=1.2, alpha=0.6)
            axes[i,j].axhline(y=0, color='black',linewidth=1.2, alpha=0.6)
            for axis in ['top','bottom','left','right']:
                axes[i,j].spines[axis].set_linewidth(0.5)
            if i == 0:
                axes[i,j].xaxis.set_label_position("top")
                axes[i,j].set_xticklabels([])
                axes[i,j].xaxis.tick_top()  # Moves the ticks to the top
                axes[i,j].xaxis.set_label_position('top')
                #axes[i,j].set_xlabel(r'$\tau$',labelpad=2, fontsize=9)
                if j == 2:
                    pass
                else:
                    axes[i,j].set_ylim(min_x, max_x)
            
            else:
                axes[i,j].set_xlabel(r'Time $(\tau/\pi)$',labelpad=2, fontsize=13)
                axes[i,j].set_xticklabels(lables_pi)
                if j == 2:
                    pass
                else:
                    axes[i,j].set_ylim(min_p, max_p)

            if j == 2:
                axes[i,j].yaxis.set_label_position("right")
                axes[i,j].yaxis.tick_right()
                #axes[i,j].invert_xaxis()
            elif j == 1:
                axes[i,j].set_yticklabels([])
    
    #################################### On Diagonal ####################################

    axes[0,0].set_title('On Diagonal', fontsize=15)

    axes[0,0].plot(GCS.t_array/(np.pi), GCS.r_JK_t[:,0,0,0], color='tab:blue')
    axes[0,0].plot(GCS.t_array/(np.pi), GCS.r_JK_t[:,1,1,0], color='tomato')
    
    axes[0,0].set_ylabel(r'Position ($x = X/x_0$)',labelpad=2, fontsize=13)

    axes[1,0].plot(GCS.t_array/(np.pi), GCS.r_JK_t[:,0,0,1], color='tab:blue')
    axes[1,0].plot(GCS.t_array/(np.pi), GCS.r_JK_t[:,1,1,1], color='tomato')
    
    axes[1,0].set_ylabel(r'Momentum ($p= P/p_0$)',labelpad=2, fontsize=13)


    #################################### Off Diagonal ####################################

    axes[0,1].set_title('Off Diagonal', fontsize=15)

    axes[0,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,0,1,0]), color='tab:blue')
    axes[0,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,1,0,0]), color='tomato')
    axes[0,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,0,1,0]),'--', color='tab:blue')
    axes[0,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,1,0,0]), '--', color='tomato')

    axes[1,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,0,1,1]), color='tab:blue')
    axes[1,1].plot(GCS.t_array/(np.pi), np.real(GCS.r_JK_t[:,1,0,1]), color='tomato')
    axes[1,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,0,1,1]), '--',color='tab:blue')
    axes[1,1].plot(GCS.t_array/(np.pi), np.imag(GCS.r_JK_t[:,1,0,1]), '--', color='tomato')
    

#################################### QRDM ####################################\

    axes[0,2].set_title(r'QRDM $(\hat{\rho}^q)$', fontsize=15)

    axes[0,2].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,2].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    axes[0,2].plot(GCS.t_array/(np.pi), GCS.C_JK_t[:,0,1], color='tab:blue')
    axes[0,2].plot(GCS.t_array/(np.pi), GCS.C_JK_t[:,1,0], ':', color='tomato')
    axes[0,2].set_ylabel(r'Contrasts $(C)$',labelpad=15, fontsize=13,rotation=-90)

    #################################### Time Evolution Phases ####################################
    axes[1,2].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,2].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    axes[1,2].plot(GCS.t_array/(np.pi), np.sin(GCS.phi_JK_t[:,0,1]), color='tab:blue')
    axes[1,2].plot(GCS.t_array/(np.pi), np.sin(GCS.phi_JK_t[:,1,0]),  color='tomato')
    
    axes[1,2].set_ylabel(r'$\cos(\phi)$',labelpad=15, fontsize=13, rotation=-90)

    
    for axis in ['top','bottom','left','right']:
        axes[0,1].spines[axis].set_linewidth(0.5)
        axes[1,1].spines[axis].set_linewidth(0.5)
        axes[1,0].spines[axis].set_linewidth(0.5)

    #################################### Plot Design ####################################
    
    axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]),  color='tab:blue', label ="Up")
    axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]),   color='tomato', label ="Down")
    axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]),  color='gray', label ="Real")
    axes[1,1].plot(np.array([-1000, -1001]),np.array([-1000, -1001]), '--',  color='gray', label ="Complex")
    
    for i in range(len(leged_lables)):
        axes[1,1].scatter(-1000,-1000, color = 'black', label = leged_lables[i])

    
    
    leg = fig.legend( loc='upper center', bbox_to_anchor=(0.50, 0.51),ncol=4+len(leged_lables), borderpad=-0.5, fontsize=11,labelspacing=0.2,handlelength=1.4)
    leg.get_frame().set_edgecolor('w')
    for line in leg.get_lines():
        line.set_linewidth(1.1)
    leg.get_frame().set_linewidth(0.4)

    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return 

def Plot_Wigner_Diag(wigner,  x_array, p_array, bar_limits, array_tick, save):
    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(1, 1, figsize=(3.5,3.5),  dpi = 300)
    fig.set_tight_layout(True)

    lables_tick = format_latex_numbers(array_tick)

    axes.set_title(r'Wigner Function $ W(\bar{r})$', fontsize=13)
    
    
    axes.axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes.axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    axes.set_ylabel(r'Momentum ($p = P/p_0$)',labelpad=2, fontsize=11)
    axes.set_xlabel(r'Position ($x = X/x_0$)',labelpad=2, fontsize=11)

    axes.set_ylim(-np.max(x_array), np.max(x_array))
    axes.set_xlim(-np.max(x_array), np.max(x_array))
    axes.set_xticks(array_tick)
    axes.set_xticklabels(lables_tick)
    axes.set_yticks(array_tick)
    axes.set_yticklabels(lables_tick)



    for axis in ['top','bottom','left','right']:
        axes.spines[axis].set_linewidth(0.5)


    #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
    #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)
    # 'RdBu'
    
    im = axes.imshow(wigner, cmap='Blues',  interpolation='bilinear', 
                        #alpha=np.abs(wigner)/np.max(wigner), 
                        origin='lower', vmin=-bar_limits[0], vmax=bar_limits[1],
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    
    #cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])   cax=cbar_ax , 
    bar = fig.colorbar(im, fraction=0.04, orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([bar_limits[0],0,bar_limits[1]])
    bar.set_ticklabels(['{:.1f}'.format(x) for x in bar.get_ticks()])
    bar.set_label(r'$ W(\tilde{r})$',fontsize=10,labelpad=-5, y = 0.4,ha='right', rotation=-90)
    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, im

def Plot_Wigner_4_Times(wigner_array, x_array, p_array, save):
    
    fig, axes = plt.subplots(2, 2, figsize=(3,3), dpi = 500)#,layout="compressed")
    
    colors = [(1, 1, 1, 0)  , (171/255, 219/255, 227/255, 1),      (236/255, 112/255, 49/255, 1)]
    
    ax = axes.flatten()
    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    titles = [r'$t=0,t_f$', r'$t_+$',r'$t_+ +t_-$', r'$t_{max}$']
    print([np.min(x_array), np.max(x_array), np.min(p_array), np.max(p_array)])
    
    # Create the heatmap
    for i in range(len(ax)):
        im = ax[i].matshow(wigner_array[i], cmap='Blues', zorder=5, interpolation='bilinear', alpha=wigner_array[i]/np.max(wigner_array[i]),origin='lower',
                          aspect=len(x_array)/len(p_array),extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])

        
        #ax[i].grid(linewidth=0.2, zorder=0)
        
        title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
        ax[i].text(0.08, 0.95, titles[i], transform=ax[i].transAxes, fontsize=8,
                verticalalignment='top', bbox=title_box)
        
        ax[i].set_xticks([-30,-15,0,15,30], minor=False)
        ax[i].set_yticks([-10,0,10], minor=False)
        ax[i].xaxis.tick_bottom()
        
       
        if i >1:
            ax[i].set_xlabel(r'$x $',labelpad=-2)
            ax[i].set_xticklabels([r'$-30$',r'$-15$',r'$0$',r'$15$',r'$30$'], fontsize=6.5)
        else:
            ax[i].set_xticklabels([])
            
            
        if i == 0 or i == 2:
            ax[i].set_ylabel(r'$ p$',labelpad=-4)
            ax[i].set_yticklabels([r'$-10$',r'$0$',r'$10$'], fontsize=6.5)
        else:
            ax[i].set_yticklabels([], minor=False)
        
        ax[i].set_axisbelow(True)

        ax[i].xaxis.set_tick_params(width=0.5,direction="in",length=3)
        ax[i].yaxis.set_tick_params(width=0.5,direction="in",length=3)
            # change all spines
        for axis in ['top','bottom','left','right']:
            ax[i].spines[axis].set_linewidth(0.5)
            
    cbar_ax = fig.add_axes([0.93, 0.18, 0.03, 0.63])
    
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical',alpha=np.linspace(0,1,100))
    bar.set_ticks([])
    bar.set_label(r'$ W(r,t)$',fontsize=8.5,labelpad=12, y=0.42, ha='right', rotation=-90)
    #ax.set_xlim(3*np.pi,9/2*np.pi)
    #ax.set_ylim(-0.05, 0.55)

    plt.subplots_adjust(wspace=0.1, hspace=-0.27)
    #fig.suptitle('(c)',x=0.06,y=0.83, fontsize=12)
    
    #a
    if save == False:
        return 
    else: 
        plt.savefig(save, bbox_inches='tight')
        return

def Plot_Wigner_Fringes(wigner_PMS,  x_array, p_array, bar_limits, array_tick, save):

    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(1, 2, figsize=(4.5,3.3),  dpi = 400)
    fig.subplots_adjust(wspace=0.1)

    #fig.suptitle('Wigner Functions of Post Measurament States')

        #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
        #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)
        # 'RdBu'
    
    im1 = axes[0].imshow(wigner_PMS[0], cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=bar_limits[0], vmax=bar_limits[1],
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    im2 = axes[1].imshow(wigner_PMS[1], cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=bar_limits[0], vmax=bar_limits[1],
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.23, 0.03, 0.5])
    bar = fig.colorbar(im2, cax=cbar_ax , orientation='vertical')#, alpha=np.linspace(0,1,100))
    bar.set_ticks([bar_limits[0],0,bar_limits[1]])
    bar.set_ticklabels(['{:.1f}'.format(x) for x in bar.get_ticks()])
    bar.set_label(r'$ W(\tilde{r})$',fontsize=10,labelpad=10, y=0.45, ha='right', rotation=-90)

    lables_tick = format_latex_numbers(array_tick)

    for i in range(2):
        
        axes[i].set_ylim(-np.max(x_array), np.max(x_array))
        axes[i].set_xlim(-np.max(x_array), np.max(x_array))
        
        axes[i].set_xticks(array_tick)
        axes[i].set_xticklabels(lables_tick)
        axes[i].set_yticks(array_tick)
        
        axes[i].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
        axes[i].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

        axes[i].set_xlabel(r'Position ($x = X/x_0$)',labelpad=2, fontsize=11)


        for axis in ['top','bottom','left','right']:
            axes[i].spines[axis].set_linewidth(0.5)
    
    axes[1].set_yticklabels([])
    axes[0].set_yticklabels(lables_tick)

    axes[0].set_title(r'$ W_+(\tilde{r})$')
    axes[1].set_title(r'$ W_-(\tilde{r})$')
    axes[0].set_ylabel(r'Momentum ($p = P/p_0$)',labelpad=0, fontsize=11)

    fig.suptitle('(b) Wigner Functions of Post-Measurement States', y=0.85)
    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight', transparent=True)
    return fig, im1, im2

def Animate_Diagonal(wigner_t,  x_array, p_array, bar_limits, array_tick, n_frames, save):

    fig, im = Plot_Wigner_Diag(wigner_t[0],  x_array, p_array, bar_limits, array_tick, False)

    def update(frame):
        """Update function for the animation."""
        # Update colormap data
        im.set_array(wigner_t[frame])

        return [im]
    
    fig.tight_layout()
    
    
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save == False:
        pass
    else:
        ani.save(filename=save,writer="pillow")
    return

def Animate_Fringes(wigner_PMS,  x_array, p_array, bar_limits, array_tick, n_frames, save):

    fig, im1,im2 = Plot_Wigner_Fringes(wigner_PMS[0],  x_array, p_array, bar_limits, array_tick, False)

    def update(frame):
        """Update function for the animation."""
        # Update colormap data
        im1.set_array(wigner_PMS[frame,0])
        im2.set_array(wigner_PMS[frame,1])

        return [im1, im2]
    
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save == False:
        pass
    else:
        ani.save(filename=save,writer="pillow")
    return

def generate_pi_ticks(maximum, step):
    # Generate numerical array
    x_vals = np.arange(0, (maximum + 1) * step * np.pi, step * np.pi)
    x_vals = np.round(x_vals, decimals=10)  # Avoid floating point issues

    labels = []
    for i in range(len(x_vals)):
        coeff = i
        if step == 1:
            label = r'$' + (f'{coeff}' if coeff > 1 else '') + r'\pi$' if coeff > 0 else r'$0$'
        elif step > 1:
            step_int = int(step) if step.is_integer() else step
            label = r'$' + (f'{coeff}' if coeff > 1 else '') + f'{step_int}\\pi$' if coeff > 0 else r'$0$'
        else:  # step < 1
            denom = int(1 / step) if (1 / step).is_integer() else f'1/{step}'
            if coeff == 0:
                label = r'$0$'
            else:
                label = r'$' + (f'{coeff}' if coeff > 1 else '') + r'\pi/' + f'{int(1/step)}$'
    # Special case: for fractional steps, show e.g. r'$2\pi/3$' rather than r'$2\pi/3$' as is.
                label = r'$' + (f'{coeff}\\pi/{int(1/step)}$' if coeff > 0 else '0')
        labels.append(label)

    return x_vals, labels

def Plot_Phase_Space_Diagonal_Total(t_array, quantum_state):

    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    fig = plt.figure(figsize=(4.5,4.5), dpi = 400)
    
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1]) 



    axes = plt.subplots(2, 2, figsize=(4.5,4.5), height_ratios=[1,2.5],width_ratios=[1, 2.5], dpi = 400)


    
    axes[0,0].axis('off')
    
    #axes[0,0].set_xticks([], minor=False)
    #axes[0,0].set_xticklabels([], fontsize=8)
    axes[0,0].set_xlim(0.1,0.2)
    axes[0,0].set_ylim(0.1,0.2)
    for i in range(len(leged_lables)):
        axes[0,0].scatter(1,1,s=1, color='black', label=leged_lables[i])
    
    axes[0,0].legend(loc='center',fontsize = 10,handletextpad=0.1)
    
    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].set_yticks([], minor=False)
    #axes[0,0].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[0,0].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[0,0].set_xlabel(r'$ p $',labelpad=2, fontsize=9)
    axes[1,0].set_xlabel(r'$x$' ,labelpad=6, fontsize=10)
    axes[1,0].set_xticks([0,0.1,0.2], minor=False)
    axes[1,0].set_xticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[1,0].set_xlim(-0.05,0.25)
    axes[1,0].xaxis.set_label_position("top")
    axes[1,0].invert_xaxis()

    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].set_xticks([], minor=False)
    #axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[1,1].set_ylabel(r'$ p $',labelpad=2, fontsize=9)
    axes[0,1].set_ylabel(r'$ | \psi(x)|^2 $' ,labelpad=14, fontsize=10,rotation=-90)
    axes[0,1].set_yticks([0,0.1,0.2], minor=False)
    axes[0,1].set_yticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[0,1].set_ylim(-0.05,0.25)
    axes[0,1].yaxis.set_label_position("right")
    
    axes[1,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    
    axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    

    axes[1,1].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    
    axes[1,1].set_ylabel(r'$ p $' ,labelpad=-10, fontsize=11)
    axes[1,1].set_xlabel(r'$ x $',labelpad=7, fontsize=11)
    axes[1,1].xaxis.set_label_position("top")

    
    #ax.set_title(title, fontsize= 10)
    #x[i].set_ylabel(r'$ p$',labelpad=-4)
    #ax[i].set_yticklabels([r'$-10$',r'$0$',r'$10$'], fontsize=6.5)
    for axis in ['top','bottom','left','right']:
        axes[0,1].spines[axis].set_linewidth(0.5)
        axes[1,1].spines[axis].set_linewidth(0.5)
        axes[1,0].spines[axis].set_linewidth(0.5)
            
    axes[0,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
    #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)

   
    line_x = axes[0,1].plot(x_array, psi_x, color='tab:blue',linewidth=1.5)
    line_p = axes[1,0].plot(psi_p, x_array, color='tomato',linewidth=1.5)
    im = axes[1,1].imshow(wigner_array, cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=-0.5, vmax=0.5,
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    cbar_ax = fig.add_axes([0.869, 0.13, 0.03, 0.45])
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([-0.5,0,0.5])
    bar.set_ticklabels([r'$-0.5$',r'$0$',r'$0.5$'],fontsize=9)
    bar.set_label(r'$ W(r,t)$',fontsize=10,labelpad=1, y=0.42, ha='right', rotation=-90)
    
    return fig, im, line_x, line_p

    return 

def plot_paper_1st(rt, t_array, t_swich, t_swich_indx, steps, save):
    
    fig, ax = plt.subplots(1,1, figsize=(3,1.7), dpi = 500)
    
    colors_green = ['lightgreen', 'limegreen' , 'forestgreen', 'green', 'darkgreen']
    colors_blue = ['lightskyblue','cornflowerblue', 'royalblue', 'blue','darkblue']
    colors_red = ['salmon','tomato','orangered','red','darkred']

    colors_from_wigner = [ (8/255, 49/255, 121/255), (39/255, 116/255, 184/255)  ]
    
    
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'

    markers = ["o",".", "s","^", "v","D","P"]
    labels = ["0", r"$t_0$",r"$t_0 + t_+$", r"$t_0 + t_+ + t_-$",r"$t_0 + 2 t_+ + t_-$",r"$t_0 +2  t_+ +2  t_-$",r"$2 (t_0 + t_+ + t_-)$"]
    
    labels_hamiltonians = [r"$H_+$",r"$H_-$",r"$H_+$",r"$H_-$"]

    ax.plot(rt[t_swich_indx[0]:t_swich_indx[1],0], rt[t_swich_indx[0]:t_swich_indx[1],1],'--', color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[0]:t_swich_indx[1],0], -rt[t_swich_indx[0]:t_swich_indx[1],1],'--', color=colors_red[3],linewidth=1.1)

    ax.plot(rt[t_swich_indx[1]:t_swich_indx[2],0], rt[t_swich_indx[1]:t_swich_indx[2],1], color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[1]:t_swich_indx[2],0], -rt[t_swich_indx[1]:t_swich_indx[2],1], color=colors_red[3],linewidth=1.1)

    ax.plot(rt[t_swich_indx[2]:t_swich_indx[3],0], rt[t_swich_indx[2]:t_swich_indx[3],1], '--', color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[2]:t_swich_indx[3],0], -rt[t_swich_indx[2]:t_swich_indx[3],1], '--', color=colors_red[3],linewidth=1.1)

    ax.plot(rt[t_swich_indx[3]+1:t_swich_indx[4],0], rt[t_swich_indx[3]+1:t_swich_indx[4],1], color=colors_from_wigner[1],linewidth=1.1)
    ax.plot(-rt[t_swich_indx[3]+1:t_swich_indx[4],0], -rt[t_swich_indx[3]+1:t_swich_indx[4],1], color=colors_red[3],linewidth=1.1)
    
    ax.plot(-100,-100,color=colors_from_wigner[1], label=r"$r_+$" )      
    ax.plot(-100,-100,color=colors_red[3], label=r"$r_+$" )   
    ax.plot(-100,-100,'--', color="grey",label=r"$H_+$" )      
    ax.plot(-100,-100,color="grey", label=r"$H_+$" )      
       
            #ax.scatter((-1)**j*x_t[i,0],(-1)**j*p_t[i,0], marker= markers[i], label = labels[i], color="white" ,edgecolors='black',zorder=10)
            #ax.scatter((-1)**j*x_t[i,0], (-1)**j*p_t[i,0], marker= markers[i])
        

    ax.set_axisbelow(True)

    ax.xaxis.set_tick_params(width=0.5,length=3)
    ax.yaxis.set_tick_params(width=0.5,length=3)
    ax.grid(zorder=0, linewidth=0.6)

    scale = 1.1

    ax.set_xlim(-np.max(-rt[:,0])*scale, np.max(-rt[:,0])*scale)
    ax.set_ylim(-np.max(rt[:,1])*scale, np.max(rt[:,1])*scale)
                       
    ax.set_ylabel(r'$p $', fontsize=10,  labelpad=-6)
    ax.set_xlabel(r'$x$', fontsize=10,  labelpad=-1)
    ax.set_axisbelow(True)

    ax.set_xticks([-15,-10,-5,0,5,10,15], minor=False)
    ax.set_yticks([-10,-5,0,5,10], minor=False)
    ax.xaxis.tick_bottom()
    ax.set_xlabel(r'$x $')
    ax.set_xticklabels([r'$-15$',r'$-10$',r'$-5$', r'$0$',r'$5$',r'$10$',r'$15$'], fontsize=6.5)
    ax.set_yticklabels([r'$-10$',r'$-5$', r'$0$',r'$5$',r'$10$'], fontsize=6.5)

    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)

    
    leg = fig.legend( loc='upper center', bbox_to_anchor=(0.995, 0.77), fontsize=8.5,labelspacing=0.7,handlelength=1.4)
    for line in leg.get_lines():
        line.set_linewidth(1.1)
    leg.get_frame().set_linewidth(0.4)
    #fig.suptitle('(b)',x=0.06,y=0.98, fontsize=12)
    
    if save == False:
        pass
    else:
        plt.savefig(save, bbox_inches='tight')
    return 

def plot_frame_presentation(wigner_array, psi_x, psi_p, delta_x, x_array, p_array, save):
    
    
    #im = plot_wigner_paper_1(axes[0,0], wigner_array_on, delta_x, x_array, p_array, 'On Diagonal')
    im = plot_wigner_paper_1(axes[1,1], wigner_array, delta_x, x_array, p_array, 'Off Diagonal')
    
    axes[1,1].plot(x_array, psi_x, color='tab:blue',linewidth=1.5)

    
    #ax.set_xlim(3*np.pi,9/2*np.pi)
    #ax.set_ylim(-0.05, 0.55)
    
    cbar_ax = fig.add_axes([0.92, 0.4, 0.03, 0.45])
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([-0.5,0,0.5])
    bar.set_label(r'$ W(r,t)$',fontsize=8.5,labelpad=8, y=0.42, ha='right', rotation=-90)
    #plt.subplots_adjust(wspace=0.1, hspace=-0.27)

    if save == False:
        return 
    else: 
        plt.savefig(save, bbox_inches='tight')
        return
         
def plot_wigner_paper_1(ax, wigner_array, delta_x, x_array, p_array, title):

    colors1 = ["lightgreen", "limegreen", "forestgreen","darkgreen"]
    colors2 = [ "tomato", "red", "darkred", "black"]

    im = ax.imshow(wigner_array, cmap='RdBu',  interpolation='bilinear', 
                    #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                    origin='lower', vmin=-0.5, vmax=0.5,
                    aspect=len(x_array)/len(p_array),
                    extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])

        
        #ax[i].grid(linewidth=0.2, zorder=0)

    
    #ax.grid(zorder=5)
    #fig.suptitle('(c)',x=0.06,y=0.83, fontsize=12)
    
    return im

def set_up_figure(wigner_array, psi_x, psi_p, delta_x, x_array, p_array, Nphonons, squeezing_r):

    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.family'] = 'STIXGeneral'
    
    fig, axes = plt.subplots(2, 2, figsize=(4.5,4.5), height_ratios=[1,2.5],width_ratios=[1, 2.5], dpi = 400)
    
    axes[0,0].axis('off')
    
    #axes[0,0].set_xticks([], minor=False)
    #axes[0,0].set_xticklabels([], fontsize=8)
    axes[0,0].set_xlim(0.1,0.2)
    axes[0,0].set_ylim(0.1,0.2)
    axes[0,0].scatter(1,1,s=1, color='black',label=r'$\delta x = {}$'.format(delta_x))
    axes[0,0].scatter(1,1,s=1, color='black',label=r'$N_p = {}$'.format(Nphonons))
    axes[0,0].scatter(1,1,s=1, color='black',label=r'$r_s = {}$'.format(squeezing_r))
    
    axes[0,0].legend(loc='center',fontsize = 10,handletextpad=0.1)
    
    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].set_yticks([], minor=False)
    #axes[0,0].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[0,0].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[0,0].set_xlabel(r'$ p $',labelpad=2, fontsize=9)
    axes[1,0].set_xlabel(r'$ | \psi(p)|^2 $' ,labelpad=6, fontsize=10)
    axes[1,0].set_xticks([0,0.1,0.2], minor=False)
    axes[1,0].set_xticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[1,0].set_xlim(-0.05,0.25)
    axes[1,0].xaxis.set_label_position("top")
    axes[1,0].invert_xaxis()

    axes[1,0].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,0].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].set_xticks([], minor=False)
    #axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    #axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=8)
    #axes[1,1].set_ylabel(r'$ p $',labelpad=2, fontsize=9)
    axes[0,1].set_ylabel(r'$ | \psi(x)|^2 $' ,labelpad=14, fontsize=10,rotation=-90)
    axes[0,1].set_yticks([0,0.1,0.2], minor=False)
    axes[0,1].set_yticklabels([r'$0$',r'$0.1$',r'$0.2$'], fontsize=8)
    axes[0,1].set_ylim(-0.05,0.25)
    axes[0,1].yaxis.set_label_position("right")
    
    axes[1,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[1,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)
    
    axes[1,1].set_xticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_xticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    

    axes[1,1].set_yticks([-delta_x,-delta_x/2,0,delta_x/2,delta_x], minor=False)
    axes[1,1].set_yticklabels([r'$-\delta x$',r'$- \frac{\delta x}{2}$',r'$0$',r'$\frac{\delta x}{2}$',r'$\delta x$'], fontsize=9)
    
    axes[1,1].set_ylabel(r'$ p $' ,labelpad=-10, fontsize=11)
    axes[1,1].set_xlabel(r'$ x $',labelpad=7, fontsize=11)
    axes[1,1].xaxis.set_label_position("top")

    
    #ax.set_title(title, fontsize= 10)
    #x[i].set_ylabel(r'$ p$',labelpad=-4)
    #ax[i].set_yticklabels([r'$-10$',r'$0$',r'$10$'], fontsize=6.5)
    for axis in ['top','bottom','left','right']:
        axes[0,1].spines[axis].set_linewidth(0.5)
        axes[1,1].spines[axis].set_linewidth(0.5)
        axes[1,0].spines[axis].set_linewidth(0.5)
            
    axes[0,1].axvline(x=0, color='black',linewidth=0.5, alpha=0.4)
    axes[0,1].axhline(y=0, color='black',linewidth=0.5, alpha=0.4)

    #title_box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.4)
    #axes[1,1].text(0.97, 5.8, r'$W^+(x,p)$', transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=title_box)

   
    line_x = axes[0,1].plot(x_array, psi_x, color='tab:blue',linewidth=1.5)
    line_p = axes[1,0].plot(psi_p, x_array, color='tomato',linewidth=1.5)
    im = axes[1,1].imshow(wigner_array, cmap='RdBu',  interpolation='bilinear', 
                        #alpha=np.abs(wigner_array)/np.max(wigner_array), 
                        origin='lower', vmin=-0.5, vmax=0.5,
                        aspect=len(x_array)/len(p_array),
                        extent = [np.min(x_array), np.max(x_array), np.min(p_array)*len(p_array)/len(x_array), np.max(p_array)*len(p_array)/len(x_array)])


    cbar_ax = fig.add_axes([0.869, 0.13, 0.03, 0.45])
    bar = fig.colorbar(im, cax=cbar_ax , orientation='vertical', alpha=np.linspace(0,1,100))
    bar.set_ticks([-0.5,0,0.5])
    bar.set_ticklabels([r'$-0.5$',r'$0$',r'$0.5$'],fontsize=9)
    bar.set_label(r'$ W(r,t)$',fontsize=10,labelpad=1, y=0.42, ha='right', rotation=-90)
    
    return fig, im, line_x, line_p
