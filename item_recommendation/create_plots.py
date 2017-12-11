
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import matplotlib.backends.backend_pdf
plt.rcParams['axes.xmargin'] = 0
# plt.rcParams['axes.ymargin'] = 0


def main():
    parser = argparse.ArgumentParser(description='Plot graphs of Precision@K and\
                                     NDCG@K from gen_log and save plots to file\
                                     with name and extension given.')
    parser.add_argument('-p', '--prefix', type=str,
                        help='prefix of the file to save plots to.')
    parser.add_argument('-e', '--extension', type=str, help='extension of saved\
                        file(.png, .tiff, .pdf, etc.).')
    parser.add_argument('-v', '--verbose', help='set flag to show plots\
                        in output.', action='store_true')
    parser.add_argument('--dpi', type=int, default=1200,
                        help='dpi of saved plots (default is 1200 - very high).')
    args = parser.parse_args()

    if not (args.prefix or args.extension):
        print('Did not specify file prefix or extension!\n')
        parser.print_help()
        sys.exit()

    gen_model_output = 'gen_log.txt'
    results = [[] for i in range(15)]
    epochs_num = 0
    with open(gen_model_output) as f:
        for line in f:
            epoch, p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10 = line.strip().split('\t')
            results[int(epoch)].append(
                [p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])
    # content
    results = np.asarray(results)
    results = results[:, -1, :]

    results_p3 = results[:, 0]
    results_p5 = results[:, 1]
    results_p10 = results[:, 2]
    results_ndcg3 = results[:, 3]
    results_ndcg5 = results[:, 4]
    results_ndcg10 = results[:, 5]

    fig = plt.figure(figsize=plt.figaspect(0.3), dpi=args.dpi)
    ax = fig.add_subplot(1, 3, 1)
    ax.set_ylabel("Precision @ 3")
    ax.set_xlabel("Training Epoch")
    ax.plot(results_p3, color='blue')
    ax.grid(True, which='both', ls='dotted')
    ax.set_aspect(1. / ax.get_data_ratio())

    ax = fig.add_subplot(1, 3, 2)
    ax.set_ylabel("Precision @ 5")
    ax.set_xlabel("Training Epoch")
    ax.plot(results_p5, color='blue')
    ax.grid(True, which='both', ls='dotted')
    ax.set_aspect(1. / ax.get_data_ratio())

    ax = fig.add_subplot(1, 3, 3)
    ax.set_ylabel("Precision @ 10")
    ax.set_xlabel("Training Epoch")
    ax.plot(results_p10, color='blue')
    ax.grid(True, which='both', ls='dotted')
    ax.set_aspect(1. / ax.get_data_ratio())

    plt.tight_layout(pad=1.0,  w_pad=2.5, h_pad=3.0)
    precision_file = args.prefix + '_p-at-k_' + args.extension
    plt.savefig(precision_file)
    if args.verbose:
        plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.3), dpi=args.dpi)
    ax = fig.add_subplot(1, 3, 1)
    ax.set_ylabel("NDCG @ 3")
    ax.set_xlabel("Training Epoch")
    ax.plot(results_ndcg3, color='blue')
    ax.grid(True, which='both', ls='dotted')
    ax.set_aspect(1. / ax.get_data_ratio())

    ax = fig.add_subplot(1, 3, 2)
    ax.set_ylabel("NDCG @ 5")
    ax.set_xlabel("Training Epoch")
    ax.plot(results_ndcg5, color='blue')
    ax.grid(True, which='both', ls='dotted')
    ax.set_aspect(1. / ax.get_data_ratio())

    ax = fig.add_subplot(1, 3, 3)
    ax.set_ylabel("NDCG @ 10")
    ax.set_xlabel("Training Epoch")
    ax.plot(results_ndcg10, color='blue')
    ax.grid(True, which='both', ls='dotted')
    ax.set_aspect(1. / ax.get_data_ratio())

    plt.tight_layout(pad=1.0,  w_pad=2.5, h_pad=3.0)
    ndcg_file = args.prefix + '_ndcg-at-k_' + args.extension
    plt.savefig(ndcg_file)
    if args.verbose:
        plt.show()

    # pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    # for f in range(1, fig.number):  # will open an empty extra figure :(
    #     pdf.savefig(f)
    # pdf.close()

# embeddings = pkl.load(open('model_dns_ori.pkl', 'rb'), encoding='latin1')
#
# embeddings = np.asanyarray(embeddings)
#
# embeddings
#
#
# print(embeddings[0].shape)
# print(embeddings[1].shape)
# print(embeddings[2].shape)


if __name__ == "__main__":
    main()
