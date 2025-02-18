import os
import hydra
import pandas as pd
import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.loss_csv_file = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'losses.csv')
        self.metric_csv_file = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'metrics.csv')

        self.losses = []
        self.metrics = []

    def log(self, message_dict):
        self.losses.append(message_dict)
        # to csv
        df = pd.DataFrame(self.losses)
        df.to_csv(self.loss_csv_file, index=False)

        # determine which keys to plot
        if 'train_episode' in df.columns:
            keys_to_plot = [col for col in df.columns if col != 'train_episode']
        else:
            keys_to_plot = df.columns.tolist()

        num_plots = len(keys_to_plot)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
        if num_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, keys_to_plot):
            if 'train_episode' in df.columns:
                df.plot(x='train_episode', y=key, ax=ax)
            else:
                df.plot(y=key, ax=ax)
                ax.set_title(key)

        plt.tight_layout()
        plt.savefig(self.loss_csv_file.replace('.csv', '_losses.png'))
        plt.close()


    def log_metrics(self, message_dict):
        # make sure the key train_epoch appears first
        print(message_dict)
        self.metrics.append(message_dict)
        # to csv
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metric_csv_file, index=False)
        # visualize and save figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        df.plot(x='train_episode', y='eval_win_rate', ax=ax1)
        ax1.set_title('Win Rate')
        
        df.plot(x='train_episode', y='eval_reward', ax=ax2)
        ax2.set_title('Reward')
        
        plt.tight_layout()
        plt.savefig(self.metric_csv_file.replace('.csv', '_metrics.png'))
        plt.close()



