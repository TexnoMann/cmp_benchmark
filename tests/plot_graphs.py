import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input", type=str, required=True,
                    help="Input pickle file for ploting graph")
parser.add_argument("-f", "--field", type=str, required=True,
                    help="Field for comparison")
parser.add_argument("-yl", "--ylabel", type=str, required=True,
                    help="Label for y axis")

FONT_SIZE=16
FIGSIZE=(10,9)

def plot_field_comparison(data, title: str, field: str, y_label: str, skip_in_y: list = None):
    field_data = data

    if not skip_in_y is None:
        f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.1}, figsize=FIGSIZE)
        sns.boxplot(data=field_data, x='algorithm',y=field,hue="planner", ax=ax_top)
        sns.boxplot(data=field_data, x='algorithm',y=field,hue="planner", ax=ax_bottom)
        ax_top.set_ylim(skip_in_y[1],160)
        ax_top.set_title(title, fontsize=FONT_SIZE)
        ax_bottom.set_ylim(0,skip_in_y[0])
        sns.despine(ax=ax_bottom)
        sns.despine(ax=ax_top, bottom=True)
        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        ax_top.grid(axis='both')
        ax_bottom.grid(axis='both')

        #remove one of the legend
        ax_top.set(xlabel=None)
        ax_top.set(ylabel=None)
        ax_top.xaxis.set_tick_params(labelsize=FONT_SIZE)
        ax_top.yaxis.set_tick_params(labelsize=FONT_SIZE)
        ax_top.legend(fontsize=FONT_SIZE)
        ax_top.legend_.set_title("Планировщик пути")
        ax_top.legend_.get_title().set_fontsize(FONT_SIZE)

        ax_bottom.legend_.remove()
        ax_bottom.set_xlabel("Алгоритм", fontsize=FONT_SIZE)
        ax_bottom.xaxis.set_tick_params(labelsize=FONT_SIZE)
        ax_bottom.yaxis.set_tick_params(labelsize=FONT_SIZE)
        ax_bottom.set(ylabel="")
        f.text(0.03, 0.55, y_label, va="center", rotation="vertical", fontsize=FONT_SIZE)
        
    else:
        f = plt.figure(figsize=FIGSIZE)
        ax = sns.boxplot(data=field_data, x='algorithm',y=field, hue="planner")
        plt.grid(axis='x')
        plt.grid(axis='y')
        plt.title(title, fontsize=FONT_SIZE)
        ax.legend(fontsize=FONT_SIZE)
        ax.legend_.set_title("Планировщик пути")
        ax.legend_.get_title().set_fontsize(FONT_SIZE)

        ax.set_xlabel("Алгоритм", fontsize=FONT_SIZE)
        ax.set_ylabel(y_label, fontsize=FONT_SIZE)
        ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
        ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    plt.show()


def plot_succes_rate_comparison(data, title: str, field: str):
    f = plt.figure(figsize=FIGSIZE)
    data['ok']*=100.0
    ax = sns.barplot(data=data, x='algorithm',y=field, hue="planner", errorbar=None)
    plt.grid(axis='x')
    plt.grid(axis='y')
    plt.title(title, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.legend_.set_title("Планировщик пути")
    ax.legend_.get_title().set_fontsize(FONT_SIZE)

    ax.set_xlabel("Алгоритм", fontsize=FONT_SIZE)
    ax.set_ylabel("Успешность планирования, %", fontsize=FONT_SIZE)
    ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
    ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    plt.show()

def main():
    args = parser.parse_args()
    print("Opening: {}".format(args.input))
    with open(args.input, 'rb') as file:
        data_plan = pickle.load(file)
        print(data_plan)
    if args.field == 'ok':
        plot_succes_rate_comparison(data_plan, "", args.field)
    else:
        plot_field_comparison(data_plan, "", args.field, args.ylabel)

        
        

if __name__ == "__main__":
    main()