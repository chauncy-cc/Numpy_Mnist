import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

color = ['blue', 'skyblue', 'green', 'red']

def plot_loss_curves(experiment_data, log_iterations, is_pytorch_version):
    # 生成图像
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    st = fig.suptitle(
        "Loss Curves for all Tasks and Hyper-parameter settings",
        fontsize = "x-large"
    )
    # 画出所有的学习曲线. i表示不同模型，j表示不同setting
    for i, results in enumerate(experiment_data):
        for j, (setting, _, _, train_loss, _) in enumerate(results):
            xs = [x * log_iterations for x in range(1, len(train_loss) + 1)]
            axes[i].plot(xs, train_loss, label='setting ' + str(j+1) + ' train_loss')
        # Prettify individual plots
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        axes[i].set_xlabel('Number of Train Iterations')
        axes[i].set_ylabel('Loss')
        axes[i].set_title('Task {}'.format(i + 1))
        axes[i].legend()
    # Prettify overall figure.
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.91)
    if is_pytorch_version:
        plt.savefig("loss_curve_pytorch.png")
    else:
        plt.savefig("loss_curve_numpy.png")
    plt.show()


def plot_accuracy_curves(experiment_data, is_pytorch_version):
    # 生成图像(设置组合个数，模型个数)
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    st = fig.suptitle(
        "Accuracy Curves for all Tasks and Hyper-parameter settings",
        fontsize="x-large"
    )
    # 画出所有的学习曲线. i表示不同模型，j表示不同setting
    for i, results in enumerate(experiment_data):
        for j, (setting, train_accuracy, _, _, _) in enumerate(results):
            xs = [x for x in range(1, len(train_accuracy) + 1)]
            axes[i].plot(xs, train_accuracy, label='setting ' + str(j+1) + ' train_accuracy')
        # Prettify individual plots
        axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        axes[i].set_xlabel('Number of Epochs')
        axes[i].set_ylabel('Accuracy')
        axes[i].set_title('Task {}'.format(i + 1))
        axes[i].legend()
    # Prettify overall figure.
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.91)
    if is_pytorch_version:
        plt.savefig("accuracy_curve_pytorch.png")
    else:
        plt.savefig("accuracy_curve_numpy.png")
    plt.show()


# 生成结果的摘要表
def plot_accuracy_summary_table(experiment_data, is_pytorch_version):
    # 填充数据
    cell_text = []
    rows = []
    setting_num = len(experiment_data[0])
    columns = ['Setting 1', 'Setting 2', 'Setting 3', 'Setting 4']
    columns = columns[:setting_num]
    for i, results in enumerate(experiment_data):
        rows.append('Model {}'.format(i + 1))
        cell_text.append([])
        for j, (setting, train_accuracy, test_accuracy, train_loss, _) in enumerate(results):
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
    plt.show()


# 生成结果的摘要表
def plot_train_time_summary_table(experiment_data, is_pytorch_version):
    # 填充数据
    cell_text = []
    rows = []
    setting_num = len(experiment_data[0])
    columns = ['Setting 1', 'Setting 2', 'Setting 3', 'Setting 4']
    columns = columns[:setting_num]
    for i, results in enumerate(experiment_data):
        rows.append('Model {}'.format(i + 1))
        cell_text.append([])
        for j, (setting, train_accuracy, test_accuracy, train_loss, train_time) in enumerate(results):
            cell_text[i].append(train_time[-1])
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
        plt.savefig("train_time_summary_table_pytorch.png")
    else:
        plt.savefig("train_time_summary_table_numpy.png")
    plt.show()