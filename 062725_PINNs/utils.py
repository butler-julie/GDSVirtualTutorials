import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cmcrameri.cm as cm
import torch

from matplotlib import animation

props = fm.FontProperties(family=['sans-serif'], size='large')
plt.rcParams['animation.writer'] = 'pillow'    
plt.rcParams['animation.writer'] = 'pillow'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.linewidth'] = 1

cmap = cm.lapaz
cmap_div = cm.berlin_r

def get_batch(t, y0, y, batch_time, batch_size):
    T, M = y.shape[:2]
    D = y.shape[-1]
    t_batch = t[:batch_time]

    c = [[i,j] for i in range(T - batch_time) for j in range(M)]
    b = [c[i] for i in np.random.choice(len(c), batch_size, replace=False)]

    for i in range(len(b)):
        if i==0:
            y0_batch = y[b[i][0], b[i][1]][None,:]
            y_batch = torch.stack([y[b[i][0]+j, b[i][1]] for j in range(batch_time)], dim=0)[:,None,:]
        else:
            y0_batch = torch.cat((y0_batch, y[b[i][0], b[i][1]][None,:]))
            y_batch = torch.cat((y_batch,
                torch.stack([y[b[i][0]+j, b[i][1]] for j in range(batch_time)], dim=0)[:,None,:]), dim=1)
    return t_batch, y0_batch, y_batch


def format_axis(ax, props, xlabel='', ylabel='', xbins=None, ybins=None):
    ax.set_xlabel(xlabel, fontproperties=props)
    ax.set_ylabel(ylabel, fontproperties=props)
    ax.yaxis.offsetText.set_fontproperties(props)
    ax.tick_params(axis='both', which='both', direction='in')
                   
    for label in ax.get_xticklabels():
        label.set_fontproperties(props)
    for label in ax.get_yticklabels():
        label.set_fontproperties(props)
        
    if xbins:
        try: ax.locator_params(axis='x', nbins=xbins)
        except: ax.locator_params(axis='x', numticks=xbins+1)
    if ybins:
        try: ax.locator_params(axis='y', nbins=ybins)
        except: ax.locator_params(axis='y', numticks=ybins+1)
            
            
def animate_node_history(model_path):
    # Set up figure axes
    fig = plt.figure(figsize=(11,5), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=5, wspace=0.15, hspace=0.3,
                          width_ratios=[1,1,0.07,0.4,1], height_ratios=[1.5,1])
    
    ax = []
    for i in range(5):
        if i != 3:
            ax.append(fig.add_subplot(gs[0,i]))
    ax.append(fig.add_subplot(gs[-1,:]))
    
    # Load saved model and history
    saved = torch.load(model_path)
    chkpt = saved['chkpt']
    loss = saved['loss']
    t_eval = saved['t_eval']
    y_true = saved['y_true']
    y_pred = saved['y_pred']
    y_eval = saved['y_eval']
    f_true = saved['f_true']
    f_pred = saved['f_pred']
    
    dt = np.diff(t_eval)[0]
    T = t_eval.max() + dt
    frames = len(loss)
    f = []
    
    # Plot true and predicted solutions
    norm = plt.Normalize(vmin=0, vmax=1)
    extent = (0, T, 0, 1)
    ax[0].imshow(y_true.T, aspect='auto', origin='lower', interpolation='nearest',
                 extent=extent, cmap=cmap, norm=norm)
    ax[0].text(0.1, 0.9, 'True', color='white', ha='left', va='top', transform=ax[0].transAxes)
    f.append(ax[1].imshow(y_pred[0].T, aspect='auto', origin='lower', interpolation='nearest',
                          extent=extent, cmap=cmap, norm=norm))
    ax[1].text(0.1, 0.9, 'Pred.', color='white', ha='left', va='top', transform=ax[1].transAxes)
    ax[1].set_yticklabels([])
    format_axis(ax[0], props, 't', 'x', ybins=3)
    format_axis(ax[1], props, 't', '', ybins=3)
    
    # Plot colorbar
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax[2])
    format_axis(ax[2], props, '', r'$\rho(t,x)$', ybins=3)
    
    # Plot true and predicted equations
    with torch.no_grad():
        ax[3].plot(y_eval, f_true, color=cmap(200), label='True')
        f.append(ax[3].plot(y_eval, f_pred[0], color=cmap(20), label='Pred.')[0])
    format_axis(ax[3], props, r'$\rho$', r'$d\rho/dt$', xbins=3, ybins=4)
    ax[3].set_ylim([f_true.min() - 0.1*f_true.max(), 1.1*f_true.max()])
    ax[3].legend(frameon=False, loc='lower center', prop=props)
    
    # Plot loss history
    f.append(ax[4].plot(chkpt*np.arange(1,2), loss[:1], color=cmap(20))[0])
    format_axis(ax[4], props, 'Iteration', 'Loss')
    ax[4].set_yscale('log')
    ax[4].set_xlim([0, chkpt*(len(loss) + 1)])
    ax[4].set_ylim([0.1*min(loss), 10*max(loss)])
     
    # Animation function
    def animate(i):
        f[0].set_data(y_pred[i+1].T)
        f[1].set_ydata(f_pred[i+1]) 
        f[2].set_data(chkpt*np.arange(1,i+2), loss[:i+1])
        return f

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=frames-1, interval=100, blit=True)
    plt.close(fig)
    
    return ani


def animate_pinn_history(model_path):
    # Set up figure axes
    fig = plt.figure(figsize=(11,5), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=6, wspace=0.15, hspace=0.3,
                          width_ratios=[1,1,1,0.07,0.1,0.07], height_ratios=[1.5,1])
    
    ax = []
    for i in range(6):
        if i != 4:
            ax.append(fig.add_subplot(gs[0,i]))
    ax.append(fig.add_subplot(gs[-1,:]))
    
    # Load saved model and history
    saved = torch.load(model_path)
    chkpt = saved['chkpt']
    loss = saved['loss']
    t_eval = saved['t_eval']
    x_eval = saved['x_eval']
    y_true = saved['y_true']
    y_pred = saved['y_pred']
    f_pred = saved['f_pred']
    
    t_eval = t_eval.reshape(y_true.shape)
    x_eval = x_eval.reshape(y_true.shape)
    
    dt = np.diff(t_eval[0])[0]
    dx = np.diff(x_eval[:,0])[0]
    T = t_eval.max() + dt
    L = x_eval.max() + dx
    frames = len(loss)
    f = []
    
    # Plot true and predicted solutions
    norm = plt.Normalize(vmin=0, vmax=1)
    extent = (0, T, 0, L)
    ax[0].imshow(y_true, aspect='auto', origin='lower', interpolation='nearest',
                 extent=extent, cmap=cmap, norm=norm)
    ax[0].text(0.1, 0.9, 'True', color='white', ha='left', va='top', transform=ax[0].transAxes)
    f.append(ax[1].imshow(y_pred[0], aspect='auto', origin='lower', interpolation='nearest',
                          extent=extent, cmap=cmap, norm=norm))
    ax[1].text(0.1, 0.9, 'Pred.', color='white', ha='left', va='top', transform=ax[1].transAxes)
    ax[1].set_yticklabels([])
    format_axis(ax[0], props, 't', 'x', ybins=3)
    format_axis(ax[1], props, 't', '', ybins=3)
    
    # Plot colorbar
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax[3])
    format_axis(ax[3], props, '', r'$\rho(t,x)$', ybins=3)
    
    # Plot PDE discrepancy
    vmax = f_pred[0].abs().max()
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    f.append(ax[2].imshow(f_pred[0], aspect='auto', origin='lower', interpolation='nearest',
                          extent=extent, cmap=cmap_div, norm=norm))
    ax[2].text(0.1, 0.9, 'PDE Diff.', color='white', ha='left', va='top', transform=ax[2].transAxes)
    ax[2].set_yticklabels([])
    format_axis(ax[2], props, 't', '', ybins=3)
    
    # Plot colorbar
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap_div, norm=norm), cax=ax[4])
    format_axis(ax[4], props, '', 'LHS - RHS', ybins=3)
    
    # Plot loss history
    f.append(ax[5].plot(chkpt*np.arange(1,2), loss[:1], color=cmap(20))[0])
    format_axis(ax[5], props, 'Iteration', 'Loss')
    ax[5].set_yscale('log')
    ax[5].set_xlim([0, chkpt*(len(loss) + 1)])
    ax[5].set_ylim([0.1*min(loss), 10*max(loss)])
     
    # Animation function
    def animate(i):
        f[0].set_data(y_pred[i+1])
        f[1].set_data(f_pred[i+1]) 
        f[2].set_data(chkpt*np.arange(1,i+2), loss[:i+1])
        return f

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=frames-1, interval=100, blit=True)
    plt.close(fig)
    
    return ani


def node_inference(ode, node, t, y0, device='cpu'):
    node.eval()

    with torch.no_grad():
        y_true = ode.solve(t, y0, device=device)
        y_pred = node.solve(t, y0, device=device)

    fig, ax = plt.subplots(1,6, figsize=(11,2.5), width_ratios=[1,1,1,0.07,0.05,0.07])
    fig.subplots_adjust(wspace=0.2)

    sm = []
    extent = (0, t.max() + np.diff(t)[0], 0, ode.L)
    ax[0].imshow(y_true.squeeze().cpu().T, aspect='auto', origin='lower', interpolation='nearest',
                 extent=extent, cmap=cmap, vmin=0, vmax=1)
    ax[0].text(0.1, 0.9, 'True', color='white', ha='left', va='top', transform=ax[0].transAxes)

    sm.append(ax[1].imshow(y_pred.squeeze().cpu().T, aspect='auto', origin='lower', interpolation='nearest',
                           extent=extent, cmap=cmap, vmin=0, vmax=1))
    ax[1].text(0.1, 0.9, 'Pred.', color='white', ha='left', va='top', transform=ax[1].transAxes)

    y_diff = y_pred - y_true
    vmax = y_diff.abs().max()
    sm.append(ax[2].imshow(y_diff.squeeze().cpu().T, aspect='auto', origin='lower', interpolation='nearest',
                           extent=extent, cmap=cmap_div, vmin=-vmax, vmax=vmax))
    ax[2].text(0.1, 0.9, 'Diff.', color='white', ha='left', va='top', transform=ax[2].transAxes)

    plt.colorbar(sm[0], cax=ax[3])
    plt.colorbar(sm[1], cax=ax[5])

    for i in range(3):
        format_axis(ax[i], props, xlabel='t', ybins=3)

    for i in range(3,len(ax)-1):
        format_axis(ax[i], props, ybins=3, ylabel=r'$\rho$')
    format_axis(ax[-1], props, ybins=3, ylabel=r'$\rho_{Pred.} - \rho_{True}$')
    
    ax[0].set_ylabel('x')
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[4].remove()
    
    return fig


def pinn_inference(ode, pinn, t, x, device='cpu'):
    pinn.to(device)
    pinn.eval()
    
    y0 = pinn.init_state(x).to(device)
    t_inf, x_inf = torch.meshgrid((t, x), indexing='xy')
    t_inf, x_inf = t_inf.reshape(-1,1), x_inf.reshape(-1,1)
    
    with torch.no_grad():
        y_true = ode.solve(t.to(device), y0[None,None,:], device).squeeze().cpu().T
        y_pred = pinn(t_inf.to(device), x_inf.to(device)).view(len(x), len(t)).cpu()

    fig, ax = plt.subplots(1,6, figsize=(11,2.5), width_ratios=[1,1,1,0.07,0.05,0.07])
    fig.subplots_adjust(wspace=0.2)

    sm = []
    extent = (0, t.max() + np.diff(t)[0], 0, ode.L)
    ax[0].imshow(y_true, aspect='auto', origin='lower', interpolation='nearest',
                 extent=extent, cmap=cmap, vmin=0, vmax=1)
    ax[0].text(0.1, 0.9, 'True', color='white', ha='left', va='top', transform=ax[0].transAxes)

    sm.append(ax[1].imshow(y_pred, aspect='auto', origin='lower', interpolation='nearest',
                           extent=extent, cmap=cmap, vmin=0, vmax=1))
    ax[1].text(0.1, 0.9, 'Pred.', color='white', ha='left', va='top', transform=ax[1].transAxes)

    y_diff = y_pred - y_true
    vmax = y_diff.abs().max()
    sm.append(ax[2].imshow(y_diff, aspect='auto', origin='lower', interpolation='nearest',
                           extent=extent, cmap=cmap_div, vmin=-vmax, vmax=vmax))
    ax[2].text(0.1, 0.9, 'Diff.', color='white', ha='left', va='top', transform=ax[2].transAxes)

    plt.colorbar(sm[0], cax=ax[3])
    plt.colorbar(sm[1], cax=ax[5])

    for i in range(3):
        format_axis(ax[i], props, xlabel='t', ybins=3)

    for i in range(3,len(ax)-1):
        format_axis(ax[i], props, ybins=3, ylabel=r'$\rho$')
    format_axis(ax[-1], props, ybins=3, ylabel=r'$\rho_{Pred.} - \rho_{True}$')
    
    ax[0].set_ylabel('x')
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[4].remove()
    
    return fig