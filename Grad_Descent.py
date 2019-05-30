from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


# f1(x,y) = x^4 + y^4 - 2* x^2 * y
# f1_bef = -24.71429 + 39.91522*x - 16.625*x^2 + 2.669192*x^3 - 0.1477273*x^4
# f1_aft = -33.28571 + 54.67569*x - 24.45833*x^2 + 4.255051*x^3 - 0.2537879*x^4

# f1(x,y) = 1/3*(f1_bef[1:3]-f1_aft[1:3])^2*x^2 ...
#         + 1/3*(f1_bef[3:5]-f1_aft[3:5])^2*y^2
#         + 1/3*(f1_bef[5:7]-f1_aft[5:7])^2*(1-x-y)^2

def get_sq_diff(f1_bef,f1_aft,l):
    sub_sq = [0,0,0]
    sub_sq[0] = (sum(f1_bef[l*0:l*1]) - sum(f1_aft[l*0:l*1]))**2
    sub_sq[1] = (sum(f1_bef[l*1:l*2]) - sum(f1_aft[l*1:l*2]))**2
    sub_sq[2] = (sum(f1_bef[l*2:l*3]) - sum(f1_aft[l*2:l*3]))**2
    return sub_sq[:]


def f1(x, y, z, sub_sq):  # x:w1, y:w2, z:w3
    return 1/3*sub_sq[0]*x**2 + 1/3*sub_sq[1]*y**2 + 1/3*sub_sq[2]*z**2


# ∇f = [∂f/∂x, ∂f/∂y]^T
def gradient_f1(xy,sub_sq):
    x = xy[0]
    y = xy[1]
    z = xy[2]

    # (Partial Difference of x, Partial Difference of y)
    return np.array([2/3*sub_sq[0]*x, 2/3*sub_sq[1]*y, 2/3*sub_sq[2]*z]);


# Gradient Descent
# init_pos = 初期位置. e.g. (x, y)
def gradient_descent_method(gradient_f, init_pos, learning_rate, sub_sq):
    eps = 1e-10

    # 計算しやすいようnumpyのarrayとする
    init_pos = np.array(init_pos)
    pos = init_pos
    pos_history = [init_pos]
    iteration_max = 1000

    # 収束するか最大試行回数に達するまで
    for i in range(iteration_max):

        # 最急上昇法の場合は-を+にする
        #pos_new = pos - learning_rate * gradient_f(pos, sub_sq)
        pos_new = pos + learning_rate * gradient_f(pos, sub_sq)

        # If x or y is negative, set to zero (projected gradient descent)
        if pos_new[0] < 0:
            pos_new[0] = 0
        if pos_new[1] < 0:
            pos_new[1] = 0
        if 1 < pos_new[1]:
            pos_new[0] = 1
        if 1 < pos_new[1]:
            pos_new[1] = 1

        # 収束条件を満たせば終了
        # np.linalg.norm(): ユークリッド距離を計算する関数
        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)

        print("[{:3d}] i = {}, f(i) = {:7f}".format(i, pos_new, f1(pos_new[0],pos_new[1],pos_new[2],sub_sq)))

    return (pos, np.array(pos_history))


# Draw raw signal
def draw_raw(fig,f1_bef,f1_aft,d):
    ax0 = fig.add_subplot(111)
    c0 = ax0.plot(d, f1_bef[:], label="Before")
    c1 = ax0.plot(d, f1_aft[:], label="After")
    ax0.set_title('Raw signal', fontsize=16)
    ax0.set_xlabel("Depth", fontsize=16)
    ax0.set_ylabel("Intensity", fontsize=16)
    ax0.legend()
    c2 = ax0.plot([3,3],[0,9], "black", linestyle="dashed")
    c3 = ax0.plot([5,5],[0,9], "black", linestyle="dashed")


# 値の高低を色で表現
def draw_color_map(fig,sub_sq):
    ax1 = fig.add_subplot(111)
    n = 256
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)
    z = np.linspace(-1.5, 1.5, n)
    X, Y, Z= np.meshgrid(x, y, z)
    T = f1(X, Y, Z, sub_sq)

    pc = ax1.pcolor(X, Y, Z, T, cmap='RdBu')

    # グラフに凡例を表示
    # http://stackoverflow.com/questions/18874135/how-to-plot-pcolor-colorbar-in-a-different-subplot-matplotlib
    # http://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # カラーバー用のaxesを用意
    fig.colorbar(pc, cax=cbar_ax)  # カラーバーを描画

    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.plot_wireframe(X, Y, Z, T, color="blue", zorder=1)
    ax2.set_xlabel("w1", fontsize=16)
    ax2.set_ylabel("w2", fontsize=16)


def draw_contour(ax, sub_sq):
    # 等高線を描く
    n = 256
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, y)

    level_num = 20
    # 等高線で同じ高さとなるエリアを色分け
    ax.contourf(X, Y, f1(X, Y, sub_sq), level_num, alpha=.75, cmap=plt.cm.hot)
    # 等高線を引く
    C = ax.contour(X, Y, f1(X, Y, sub_sq), level_num, colors='black', linewidth=.5)
    ax.clabel(C, inline=1, fontsize=10)
    ax.set_title('contour')


def is_valid(num):
    return -5 < num < 5;


def main():

    d = np.linspace(1, 7, 60)
    #Define
    f1_bef = -10.429 + 26.44*d - 12.917*d**2 + 2.2929*d**3 - 0.1364*d**4
    f1_aft = -0.1429 + 2.3225*d + 0.9167*d**2 - 0.4394*d**3 + 0.0379*d**4

    # f1_bef = -24.71429 + 39.91522*d - 16.625*d**2 + 2.669192*d**3 - 0.1477273*d**4
    # f1_aft = -33.28571 + 54.67569*d - 24.45833*d**2 + 4.255051*d**3 - 0.2537879*d**4

    l = int(np.size(d,0)/3)
    sub_sq = get_sq_diff(f1_bef, f1_aft, l)


    fig0 = plt.figure(0)
    draw_raw(fig0, f1_bef, f1_aft, d)

    # カラーマップを表示
    fig1 = plt.figure(1)
    #draw_color_map(fig1, sub_sq)

    #learning_rates = [0.001, 0.01, 0.05, 0.1]
    learning_rates = [0.0001]

    # 収束する様子を表示するためのグラフ
    fig2 = plt.figure(3)

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history = gradient_descent_method(gradient_f1, (0.1, 0.1, 0.1), learning_rate, sub_sq)

        # subplotの場所を指定
        ax = plt.subplot(2, 2, (i + 1))  # 2行2列の意味

        # 等高線を描画
        draw_contour(ax, sub_sq)

        # グラフのタイトル
        ax.set_title("learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history)))

        # 移動した点を表示
        for pos in pos_history:
            if is_valid(pos[0]) and is_valid(pos[1]):
                ax.plot(pos[0], pos[1], 'o')

        # 点同士を線で結ぶ
        for i in range(len(pos_history) - 1):
            x1 = pos_history[i][0]
            y1 = pos_history[i][1]
            x2 = pos_history[i + 1][0]
            y2 = pos_history[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

    # タイトルが重ならないようにする
    fig2.tight_layout()

    # 画像を表示
    plt.show()

    # 画像を保存
    fig0.savefig('raw_signal.png')
    fig1.savefig('color_map.png')
    fig2.savefig('2d-result.png')


    w1 = pos[0]
    w2 = pos[1]
    w3 = pos[2]
    print("x = {:7f}, y = {:7f}, z = {:7f}".format(w1, w2, w3))


main()
