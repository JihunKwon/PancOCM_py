
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle

# Square Difference for Step A (two segments)
def get_sq_diff_A(f_bef,f_aft, f_10m,l):
    sub_sq = [0,0,0,0]
    sub_sq[0] = (sum(f_bef[l*0:l*1]) - sum(f_aft[l*0:l*1]))**2  # Difference between before and after
    sub_sq[1] = (sum(f_bef[l*1:l*2]) - sum(f_aft[l*1:l*2]))**2
    sub_sq[2] = (sum(f_aft[l*0:l*1]) - sum(f_10m[l*0:l*1]))**2  # Difference between after and 10min after
    sub_sq[3] = (sum(f_aft[l*1:l*2]) - sum(f_10m[l*1:l*2]))**2
    return sub_sq[:]

# Square Difference for first half of Step B
def get_sq_diff_B1(f_bef,f_aft, f_10m,l):
    sub_sq = [0,0,0,0]
    sub_sq[0] = (sum(f_bef[l*0:l*1]) - sum(f_aft[l*0:l*1]))**2
    sub_sq[1] = (sum(f_bef[l*1:l*2]) - sum(f_aft[l*1:l*2]))**2
    sub_sq[2] = (sum(f_aft[l*0:l*1]) - sum(f_10m[l*0:l*1]))**2  # Difference between after and 10min after
    sub_sq[3] = (sum(f_aft[l*1:l*2]) - sum(f_10m[l*1:l*2]))**2
    return sub_sq[:]

# Square Difference for second half of Step B
def get_sq_diff_B2(f_bef,f_aft, f_10m,l):
    sub_sq = [0,0,0,0]
    sub_sq[0] = (sum(f_bef[l*2:l*3]) - sum(f_aft[l*2:l*3]))**2
    sub_sq[1] = (sum(f_bef[l*3:l*4]) - sum(f_aft[l*3:l*4]))**2
    sub_sq[2] = (sum(f_aft[l*2:l*3]) - sum(f_10m[l*2:l*3]))**2  # Difference between after and 10min after
    sub_sq[3] = (sum(f_aft[l*3:l*4]) - sum(f_10m[l*3:l*4]))**2
    return sub_sq[:]

def f1(x, y, sub_sq):  # x:w1, y:w2
    return 1/2*sub_sq[0]*x**2 + 1/2*sub_sq[2]*x**2 + 1/2*sub_sq[0]*y**2 + 1/2*sub_sq[3]*y**2

# ∇f = [∂f/∂x, ∂f/∂y]^T
def gradient_f1(xy,sub_sq):
    x = xy[0]
    y = xy[1]
    # (Partial Difference of x, Partial Difference of y)
    return np.array([sub_sq[0]*x + sub_sq[2]*x, sub_sq[1]*y + sub_sq[3]*y]);

# Gradient Descent
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
        if (pos_new[0]+pos_new[1]) >= 1.0:
            break
        if pos_new[0] < 0.1:
            pos_new[0] = 0.1
        if pos_new[1] < 0.1:
            pos_new[1] = 0.1
        if 0.9 < pos_new[1]:
            pos_new[0] = 0.9
        if 0.9 < pos_new[1]:
            pos_new[1] = 0.9

        # 収束条件を満たせば終了
        # np.linalg.norm(): ユークリッド距離を計算する関数
        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)

        print("[{:3d}] i = {}, f(i) = {:7f}".format(i, pos_new, f1(pos_new[0],pos_new[1],sub_sq)))
        a = np.shape(np.array(pos_history))
    return (pos, np.array(pos_history))


# Draw raw signal
def draw_raw(fig,f1_bef,f1_aft, f1_10m,d,ocm):
    ax0 = fig.add_subplot(1, 3, (ocm+1))
    c0 = ax0.plot(d, f1_bef, label="Before")
    c1 = ax0.plot(d, f1_aft, label="After")
    c1_ = ax0.plot(d, f1_10m, label="10min After")
    ax0.set_title('Raw signal, OCM{:1d}'.format(ocm), fontsize=16)
    ax0.set_xlabel("Depth", fontsize=16)
    ax0.set_ylabel("Intensity", fontsize=16)
    ax0.legend()
    #c2 = ax0.plot([4,4],[0,9], "black", linestyle="solid")
    #c2 = ax0.plot([2.5,2.5],[0,9], "black", linestyle="dashed")
    #c2 = ax0.plot([5.5,5.5],[0,9], "black", linestyle="dashed")

def draw_3D(ax,sub_sq):
    n = 20
    x = np.linspace(0, 1.0, n)
    y = np.linspace(0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    Z = f1(X, Y, sub_sq)

    # Draw in 3D
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.plot_wireframe(X, Y, Z, color="blue", zorder=1)
    ax.set_xlabel("w1", fontsize=12)
    ax.set_ylabel("w2", fontsize=12)
    ax.set_zlabel("Cost Function (Difference \n between before and after water)", fontsize=10)

    #ax.scatter(X, Y, Z, s=2, color="red", zorder=2)


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
    ax.set_xlabel("w1", fontsize=12)
    ax.set_ylabel("w2", fontsize=12)


def is_valid(num):
    return 0 < num < 1;


def main():
    with open('t012_3.pkl', 'rb') as f:
        t0, t1, t2 = pickle.load(f)

    # = int(np.size(t0,0))
    d = np.linspace(300, 300 + np.size(t0,0), np.size(t0,0))

    #for bh in range(0,np.size(t0,1)):

    f_bef = np.zeros((3,np.size(t0,0)))
    f_aft = np.zeros((3,np.size(t0,0)))
    f_10m = np.zeros((3,np.size(t0,0)))

    ######################## Step A optimization (separate to two segments) ########################
    l_2 = int(np.size(t0,0)/2)
    sub_sq_A = np.zeros((3,4))
    pos_history_A = np.zeros((3,2))
    w_a1 = [0,0,0]
    w_a2 = [0,0,0]

    # OCM0
    f_bef[0][:] = np.mean(t0[:, :, 0], 1)
    f_aft[0][:] = np.mean(t0[:, :, 1], 1)
    f_10m[0][:] = np.mean(t0[:, :, 2], 1)
    # OCM1
    f_bef[1][:] = np.mean(t1[:, :, 0], 1)
    f_aft[1][:] = np.mean(t1[:, :, 1], 1)
    f_10m[1][:] = np.mean(t1[:, :, 2], 1)
    # OCM2
    f_bef[2][:] = np.mean(t2[:, :, 0], 1)
    f_aft[2][:] = np.mean(t2[:, :, 1], 1)
    f_10m[2][:] = np.mean(t2[:, :, 2], 1)

    fig0 = plt.figure(figsize=(15, 5))

    for ocm in range(3):
        sub_sq_A[ocm][:] = get_sq_diff_A(f_bef[ocm][:], f_aft[ocm][:], f_10m[ocm][:], l_2)
        draw_raw(fig0, f_bef[ocm][:], f_aft[ocm][:], f_10m[ocm][:], d, ocm)

    # 収束する様子を表示するためのグラフ
    fig2 = plt.figure(figsize=(15,5))
    learning_rates = [0.00005]

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history_A_0 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_A[0])  # OCM0
        ans, pos_history_A_1 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_A[1])  # OCM1
        ans, pos_history_A_2 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_A[2])  # OCM2

        ### 2D Plot ###
        ## OCM0
        ocm = 0
        ax = plt.subplot(1,3,ocm+1)
        draw_contour(ax, sub_sq_A[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_A)))
        # Show points
        for pos in pos_history_A_0:
            if is_valid(pos[0]) and is_valid(pos[1]):
                ax.plot(pos[0], pos[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_A_0) - 1):
            x1 = pos_history_A_0[i][0]
            y1 = pos_history_A_0[i][1]
            x2 = pos_history_A_0[i + 1][0]
            y2 = pos_history_A_0[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)
        # Save last point
        w_a1[ocm] = pos[0]
        w_a2[ocm] = pos[1]
        print("OCM"+str(ocm)+": w_a1 = {:.4f}, w_a2 = {:.4f}, w_a1+w_a2 = {:.4f}".format(w_a1[ocm], w_a2[ocm], w_a1[ocm] + w_a2[ocm]))

        ## OCM1
        ocm = 1
        ax = plt.subplot(1,3,ocm+1)
        draw_contour(ax, sub_sq_A[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_A)))
        # Show points
        for pos in pos_history_A_1:
            if is_valid(pos[0]) and is_valid(pos[1]):
                ax.plot(pos[0], pos[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_A_1) - 1):
            x1 = pos_history_A_1[i][0]
            y1 = pos_history_A_1[i][1]
            x2 = pos_history_A_1[i + 1][0]
            y2 = pos_history_A_1[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)
        # Save last point
        w_a1[ocm] = pos[0]
        w_a2[ocm] = pos[1]
        print("OCM"+str(ocm)+": w_a1 = {:.4f}, w_a2 = {:.4f}, w_a1+w_a2 = {:.4f}".format(w_a1[ocm], w_a2[ocm], w_a1[ocm] + w_a2[ocm]))

        ## OCM2
        ocm = 2
        ax = plt.subplot(1,3,ocm+1)
        draw_contour(ax, sub_sq_A[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_A)))
        # Show points
        for pos in pos_history_A_2:
            if is_valid(pos[0]) and is_valid(pos[1]):
                ax.plot(pos[0], pos[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_A_2) - 1):
            x1 = pos_history_A_2[i][0]
            y1 = pos_history_A_2[i][1]
            x2 = pos_history_A_2[i + 1][0]
            y2 = pos_history_A_2[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)
        # Save last point
        w_a1[ocm] = pos[0]
        w_a2[ocm] = pos[1]
        print("OCM"+str(ocm)+": w_a1 = {:.4f}, w_a2 = {:.4f}, w_a1+w_a2 = {:.4f}".format(w_a1[ocm], w_a2[ocm], w_a1[ocm] + w_a2[ocm]))

        '''
        ### 3D Plot ###
        fig3d = plt.figure(2)
        ax2 = fig3d.add_subplot(111, projection='3d')
        draw_3D(ax2, sub_sq)
        #ax2.set_title("3D")
        for i in range(len(pos_history)):
            ax2.scatter(pos_history[i][0],pos_history[i][1],f1(pos_history[i][0],pos_history[i][1],sub_sq), s = 5, color = "red", zorder = 1)
        '''

    # タイトルが重ならないようにする
    fig2.tight_layout()

    # 画像を表示
    plt.show()

    # 画像を保存
    fig0.savefig('raw_signal.png')
    fig2.savefig('2d-result.png')

    ######################## Step B optimization (separate to two more segments) ########################
    l_4 = int(np.size(t0,0)/4)
    sub_sq_B1 = np.zeros((3,4))
    sub_sq_B2 = np.zeros((3,4))
    pos_history_B1 = np.zeros((3,2))
    pos_history_B2 = np.zeros((3,2))
    w_b1_1 = [0,0,0]
    w_b1_2 = [0,0,0]
    w_b2_1 = [0,0,0]
    w_b2_2 = [0,0,0]

    for ocm in range(3):
        sub_sq_B1[ocm][:] = get_sq_diff_B1(f_bef[ocm][:], f_aft[ocm][:], f_10m[ocm][:], l_4)
        sub_sq_B2[ocm][:] = get_sq_diff_B2(f_bef[ocm][:], f_aft[ocm][:], f_10m[ocm][:], l_4)

    fig4 = plt.figure(figsize=(15, 5)) # Graph to draw convergence

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history_B1_0 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B1[0])
        ans, pos_history_B1_1 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B1[1])
        ans, pos_history_B1_2 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B1[2])
        ans, pos_history_B2_0 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B2[0])
        ans, pos_history_B2_1 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B2[1])
        ans, pos_history_B2_2 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B2[2])

        ### 2D Plot ###
        ## OCM0, 1st Segment
        ocm = 0
        ax = plt.subplot(2,3,ocm+1)
        draw_contour(ax, sub_sq_B1[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B1)))
        # Show points
        for posB1 in pos_history_B1_0:
            if is_valid(posB1[0]) and is_valid(posB1[1]):
                ax.plot(posB1[0], posB1[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_B1_0) - 1):
            x1 = pos_history_B1_0[i][0]
            y1 = pos_history_B1_0[i][1]
            x2 = pos_history_B1_0[i + 1][0]
            y2 = pos_history_B1_0[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        ## OCM0, 2nd Segment
        ax = plt.subplot(2,3,ocm+4)
        draw_contour(ax, sub_sq_B2[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B2)))
        # Show points
        for posB2 in pos_history_B2_0:
            if is_valid(posB2[0]) and is_valid(posB2[1]):
                ax.plot(posB2[0], posB2[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_B2_0) - 1):
            x1 = pos_history_B2_0[i][0]
            y1 = pos_history_B2_0[i][1]
            x2 = pos_history_B2_0[i + 1][0]
            y2 = pos_history_B2_0[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        # Save last point
        w_b1_1[ocm] = posB1[0] * w_a1[ocm]
        w_b1_2[ocm] = posB1[1] * w_a1[ocm]
        w_b2_1[ocm] = posB2[0] * w_a2[ocm]
        w_b2_2[ocm] = posB2[1] * w_a2[ocm]
        print("OCM"+str(ocm)+": w_b1_1 = {:.4f}, w_b1_2 = {:.4f}, w_b2_1 = {:.4f}, w_b2_2 = {:.4f}, All = {:.4f}"
              .format(w_b1_1[ocm], w_b1_2[ocm], w_b2_1[ocm], w_b2_2[ocm], w_b1_1[ocm]+w_b1_2[ocm]+w_b2_1[ocm]+w_b2_2[ocm]))




        ## OCM1, 1st Segment
        ocm = 1
        ax = plt.subplot(2,3,ocm+1)
        draw_contour(ax, sub_sq_B1[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B1)))
        # Show points
        for posB1 in pos_history_B1_1:
            if is_valid(posB1[0]) and is_valid(posB1[1]):
                ax.plot(posB1[0], posB1[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_B1_1) - 1):
            x1 = pos_history_B1_1[i][0]
            y1 = pos_history_B1_1[i][1]
            x2 = pos_history_B1_1[i + 1][0]
            y2 = pos_history_B1_1[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        ## OCM1, 2nd Segment
        ax = plt.subplot(2,3,ocm+4)
        draw_contour(ax, sub_sq_B2[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B2)))
        # Show points
        for posB2 in pos_history_B2_1:
            if is_valid(posB2[0]) and is_valid(posB2[1]):
                ax.plot(posB2[0], posB2[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_B2_1) - 1):
            x1 = pos_history_B2_1[i][0]
            y1 = pos_history_B2_1[i][1]
            x2 = pos_history_B2_1[i + 1][0]
            y2 = pos_history_B2_1[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        # Save last point
        w_b1_1[ocm] = posB1[0] * w_a1[ocm]
        w_b1_2[ocm] = posB1[1] * w_a1[ocm]
        w_b2_1[ocm] = posB2[0] * w_a2[ocm]
        w_b2_2[ocm] = posB2[1] * w_a2[ocm]
        print("OCM"+str(ocm)+": w_b1_1 = {:.4f}, w_b1_2 = {:.4f}, w_b2_1 = {:.4f}, w_b2_2 = {:.4f}, All = {:.4f}"
              .format(w_b1_1[ocm], w_b1_2[ocm], w_b2_1[ocm], w_b2_2[ocm], w_b1_1[ocm]+w_b1_2[ocm]+w_b2_1[ocm]+w_b2_2[ocm]))



        ## OCM2, 1st Segment
        ocm = 2
        ax = plt.subplot(2,3,ocm+1)
        draw_contour(ax, sub_sq_B1[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B1)))
        # Show points
        for posB1 in pos_history_B1_2:
            if is_valid(posB1[0]) and is_valid(posB1[1]):
                ax.plot(posB1[0], posB1[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_B1_2) - 1):
            x1 = pos_history_B1_2[i][0]
            y1 = pos_history_B1_2[i][1]
            x2 = pos_history_B1_2[i + 1][0]
            y2 = pos_history_B1_2[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        ## OCM2, 2nd Segment
        ax = plt.subplot(2,3,ocm+4)
        draw_contour(ax, sub_sq_B2[ocm])
        ax.set_title("OCM"+ str(ocm) + ", learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B2)))
        # Show points
        for posB2 in pos_history_B2_2:
            if is_valid(posB2[0]) and is_valid(posB2[1]):
                ax.plot(posB2[0], posB2[1], 'o')
        # Connect points with lines
        for i in range(len(pos_history_B2_2) - 1):
            x1 = pos_history_B2_2[i][0]
            y1 = pos_history_B2_2[i][1]
            x2 = pos_history_B2_2[i + 1][0]
            y2 = pos_history_B2_2[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        # Save last point
        w_b1_1[ocm] = posB1[0] * w_a1[ocm]
        w_b1_2[ocm] = posB1[1] * w_a1[ocm]
        w_b2_1[ocm] = posB2[0] * w_a2[ocm]
        w_b2_2[ocm] = posB2[1] * w_a2[ocm]
        print("OCM"+str(ocm)+": w_b1_1 = {:.4f}, w_b1_2 = {:.4f}, w_b2_1 = {:.4f}, w_b2_2 = {:.4f}, All = {:.4f}"
              .format(w_b1_1[ocm], w_b1_2[ocm], w_b2_1[ocm], w_b2_2[ocm], w_b1_1[ocm]+w_b1_2[ocm]+w_b2_1[ocm]+w_b2_2[ocm]))



        # タイトルが重ならないようにする
        fig4.tight_layout()

        # 画像を表示
        plt.show()

        # 画像を保存
        fig4.savefig('2d-result_step2.png')



print("End of code")

main()
