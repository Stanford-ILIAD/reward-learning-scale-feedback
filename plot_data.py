from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from config import CFG
import copy

matplotlib.rcParams["legend.frameon"] = False
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

plt.rc('legend', fontsize=16)  # using a size in points

error_labels = {'reward': 'Relative Reward',
                'alignment': 'Alignment',
                'loglikelihood': 'Log-Likelihood',
                'slider': 'Slider'
                }

path = CFG['path']
sigma = CFG['sigma_plot']

def main():
    df = pd.DataFrame()
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if "csv" in file and "_cropped" not in file:
            print('file', file)
            df_file = pd.read_csv(path + "/" + file, index_col=0)
            df = pd.concat([df, df_file])


    df = df.loc[df['sigma'] == sigma]
    df = df.loc[df['$alpha$'] > 0.01]
    df = replace_solver_labels(df)

    measures_plot_pairs(df, 'alignment', sigma)
    measures_plot_pairs(df, 'reward', sigma)
    measures_plot_pairs(df, 'loglikelihood', sigma)
    # slider_historgram(df, sigma)
    plt.show()


def replace_solver_labels(df):
    relabel = {'random_res:0.1': 'Random - Scale',
               'regret_res:0.1': 'MaxRegret - Scale',
               'information_res:0.1': 'Information - Scale',
               'random_res:1.0': 'Random - Soft Choice',
               'regret_res:1.0': 'MaxRegret - Soft Choice',
               'information_res:1.0': 'Information - Soft Choice',
               'random_res:2.0': 'Random - Strict Choice',
               'regret_res:2.0': 'MaxRegret - Strict Choice',
               'information_res:2.0': 'Information - Strict Choice'
               }
    for old_label in relabel.keys():
        df['solver'] = df['solver'].replace([old_label], relabel[old_label])
    return df

def alpha_plots(df, measure='Alignment', sigma=0.1):
    """

    :param df:
    :param hueorder:
    :param measure:
    :param sigma:
    :return:
    """
    hueorder = ['Scale', 'Choice']
    alphas = [0.25, 0.5, 0.75, 1.0]
    colors = ["#f46513", "#f46513", "#f46513",
              "#0d58f8", "#0d58f8", "#0d58f8"]
    solvers = ['Information', 'MaxRegret', 'Random']
    fig, axes = plt.subplots(nrows=len(solvers), ncols=len(alphas), figsize=(18, 9), sharex=True, sharey=True)
    axes[0][0].set_ylabel(error_labels[measure], fontsize=18)
    for solver_idx in range(len(solvers)):
        for alpha_idx in range(len(alphas)):
            ax = axes[solver_idx][alpha_idx]
            sns.set_palette(sns.color_palette([colors[0], colors[3], '#b8b8b8']))
            df_tmp = copy.deepcopy(df)
            df_tmp = df_tmp.loc[df_tmp['$alpha$'] == alphas[alpha_idx]]
            # df_tmp = df_tmp.loc[df_tmp['solver'] == solvers[solver_idx]]
            print('plot', alpha_idx, solver_idx)
            #  Relabel Data for the solver currently plotted
            df_tmp['solver'] = df_tmp['solver'].replace([solvers[solver_idx] + ' - Scale'], 'Scale')
            df_tmp['solver'] = df_tmp['solver'].replace([solvers[solver_idx] + ' - Soft Choice'], 'Choice')
            df_tmp['solver'] = df_tmp['solver'].replace([solvers[solver_idx] + ' - Strict Choice'], 'Strict Choice')

            sns.lineplot(ax=ax, x='iter', y=measure, hue='solver', hue_order=hueorder, data=df_tmp, ci=90)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            if solver_idx ==2:
                ax.set_xlabel("Iteration\n"+"$\\alpha=$"+str(alphas[alpha_idx]), fontsize=22)
            if alpha_idx == 0:
                ax.set_ylabel(solvers[solver_idx]+"\n"+error_labels[measure], fontsize=22)
            ax.get_legend().set_visible(False)
            handles, labels = ax.get_legend_handles_labels()

            if alpha_idx == 0:
                ax.legend(handles, labels, loc='lower left', fontsize=20)

    fig.tight_layout()
    plt.savefig(path + "alpha_plot_" + measure + "_sigma" + str(int(sigma * 10)))


def measures_plot_pairs(df, measure='alignment', sigma=0.1):
    """

    :param df:
    :param hueorder:
    :return:
    """
    solvers = ['Information', 'MaxRegret', 'Random']
    fig, axes = plt.subplots(nrows=1, ncols=len(solvers), figsize=(8, 4), sharey=True)

    colors = ["#f46513", "#f46513", "#f46513",
              "#0d58f8", "#0d58f8", "#0d58f8"]
    plt.rcParams["legend.loc"] = 'lower right'
    for idx in range(len(solvers)):
        sns.set_palette(sns.color_palette([colors[idx], colors[idx + 3], '#b8b8b8']))
        df_tmp = copy.deepcopy(df)
        #  Relabel Data for the solver currently plotted
        df_tmp['solver'] = df_tmp['solver'].replace([solvers[idx] + ' - Scale'], 'Scale')
        df_tmp['solver'] = df_tmp['solver'].replace([solvers[idx] + ' - Soft Choice'], 'Choice')
        df_tmp['solver'] = df_tmp['solver'].replace([solvers[idx] + ' - Strict Choice'], 'Strict Choice')

        hueorder = ['Scale', 'Choice']
        sns.lineplot(ax=axes[idx], x='iter', y=measure, hue='solver', hue_order=hueorder, data=df_tmp, ci=90)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].set_xlabel("Iteration", fontsize=22)
        axes[idx].tick_params(axis='x', labelsize=20)
        axes[idx].tick_params(axis='y', labelsize=20)
        axes[idx].set_xlabel("Iteration", fontsize=22)
        axes[idx].set_ylabel(error_labels[measure], fontsize=22)
        axes[idx].title.set_text(solvers[idx])
        axes[idx].get_legend().set_visible(False)
        handles, labels = axes[idx].get_legend_handles_labels()

        if idx == 0:
            axes[idx].legend(handles, labels, loc='lower left', fontsize=20)
            # axes[idx].legend().set_title('') bbox_to_anchor=(1,0),loc='lower right',

        if measure != 'loglikelihood':
            lb = .3 if measure == 'alignment' else .7
            axes[idx].set_ylim([lb, 1])
        else:
            axes[idx].set_ylim([-8, 0])

    fig.tight_layout()
    plt.savefig(path + "plot_pairs_" + str(int(sigma * 10)) + "_" + str(measure))


def slider_historgram(df, hueorder, sigma=0.1):
    """

    :param df:
    :param hueorder:
    :return:
    """
    df = df.loc[df['iter'] <= 5]
    df = df.loc[df['iter'] > 0]
    fig, axes = plt.subplots(nrows=1, ncols=len(hueorder), figsize=(8, 3), sharey=True)

    hueorder = ['Random - Scale', 'MaxRegret - Scale', 'Information - Scale']
    for idx in range(len(axes)):
        df_filtered = df.loc[df['solver'] == hueorder[idx]]
        sns.histplot(ax=axes[idx], x='slider', data=df_filtered, hue='solver', bins=21)
        axes[idx].title.set_text(hueorder[idx])
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].tick_params(axis='x', labelsize=16)
        axes[idx].tick_params(axis='y', labelsize=16)
        axes[idx].set_xlabel("Slider Position", fontsize=18)

        axes[idx].legend().set_title('')
        if idx > 0:
            axes[idx].get_legend().set_visible(False)
    fig.tight_layout()
    plt.savefig(path + "slider_histogram" + str(int(sigma * 10)) + str(hueorder))


if __name__ == "__main__":
    main()
