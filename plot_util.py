import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_loss_curves(experiment_data, log_iterations, is_run_on_server, is_pytorch_version):
    # 生成图像
    fig, axes = plt.subplots(3, 2, figsize=(22, 12))
    st = fig.suptitle(
        "Loss Curves for all Tasks and Hyper-parameter settings",
        fontsize = "x-large"
    )
    # 画出所有的学习曲线. i表示不同模型，j表示不同setting
    for i, results in enumerate(experiment_data):
        for j, (setting, _, _, train_loss) in enumerate(results):
            # Plot.
            xs = [x * log_iterations for x in range(1, len(train_loss) + 1)]
            axes[j, i].plot(xs, train_loss, label='train_loss')
            # Prettify individual plots
            axes[j, i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            axes[j, i].set_xlabel('Number of Train Iterations')
            axes[j, i].set_ylabel('Epochs: {}, Learning rate: {}. Loss'.format(*setting))
            axes[j, i].set_title('Task {}'.format(i + 1))
            axes[j, i].legend()
        # Prettify overall figure.
        plt.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.91)
        if is_pytorch_version:
            plt.savefig("loss_curve_pytorch.png")
        else:
            plt.savefig("loss_curve_numpy.png")
        if not is_run_on_server:
            plt.show()


def plot_accuracy_curves(experiment_data, is_run_on_server, is_pytorch_version):
    # 生成图像
    fig, axes = plt.subplots(3, 2, figsize=(22, 12))
    st = fig.suptitle(
        "Accuracy Curves for all Tasks and Hyper-parameter settings",
        fontsize="x-large"
    )
    # 画出所有的学习曲线. i表示不同模型，j表示不同setting
    for i, results in enumerate(experiment_data):
        for j, (setting, train_accuracy, test_accuracy, _) in enumerate(results):
            # Plot.
            xs = [x for x in range(1, len(train_accuracy) + 1)]
            axes[j, i].plot(xs, train_accuracy, label='train_accuracy')
            axes[j, i].plot(xs, test_accuracy, label='test_accuracy')
            # Prettify individual plots
            axes[j, i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            axes[j, i].set_xlabel('Number of Epochs')
            axes[j, i].set_ylabel('Epochs: {}, Learning rate: {}. Accuracy'.format(*setting))
            axes[j, i].set_title('Task {}'.format(i+1))
            axes[j, i].legend()
        # Prettify overall figure.
        plt.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.91)
        if is_pytorch_version:
            plt.savefig("accuracy_curve_pytorch.png")
        else:
            plt.savefig("accuracy_curve_numpy.png")
        if not is_run_on_server:
            plt.show()



# 生成结果的摘要表
def plot_summary_table(experiment_data, is_run_on_server, is_pytorch_version):
    # 填充数据
    cell_text = []
    rows = []
    columns = ['Setting 1', 'Setting 2', 'Setting 3']
    for i, results in enumerate(experiment_data):
        rows.append('Model {}'.format(i + 1))
        cell_text.append([])
        for j, (setting, train_accuracy, test_accuracy, train_loss) in enumerate(results):
            cell_text[i].append(test_accuracy[-1])
    # 生成表
    fig = plt.figure(frameon=False)
    ax = plt.gca()
    the_table = ax.table(
        cellText = cell_text,
        rowLabels = rows,
        colLabels = columns,
        loc = 'center'
    )
    the_table.scale(1, 4)
    # Prettify.
    ax.patch.set_facecolor('None')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if is_pytorch_version:
        plt.savefig("accuracy_summary_table_pytorch.png")
    else:
        plt.savefig("accuracy_summary_table_numpy.png")
    if not is_run_on_server:
        plt.show()


# 生成训练时间的摘要表
def plot_train_time_table():
    pass