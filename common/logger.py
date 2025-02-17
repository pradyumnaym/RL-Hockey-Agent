import pandas as pd
import matplotlib.pyplot as plt
class Logger:
    def __init__(self, config):
        self.config = config
        self.loss_csv_file = config['out_folder'] + '/loss.csv'
        self.metric_csv_file = config['out_folder'] + '/metrics.csv'

        self.losses = []
        self.metrics = []

    def log(self, message_dict):
        print(message_dict)
        self.losses.append(message_dict)
        # to csv
        df = pd.DataFrame(self.losses)
        df.to_csv(self.loss_csv_file, index=False)

        # visualize and save figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        df.plot(x='train_episode', y='critic_loss', ax=ax1)
        ax1.set_title('Critic Loss')
        
        df.plot(x='train_episode', y='actor_loss', ax=ax2) 
        ax2.set_title('Actor Loss')
        
        df.plot(x='train_episode', y='alpha_loss', ax=ax3)
        ax3.set_title('Alpha Loss')
        
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
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
        
        if 'eval_win_rate_weak' in df.columns:
            df.plot(x='train_episode', y='eval_win_rate_weak', ax=ax1)
            ax1.set_title('Win Rate Weak')

        if 'eval_win_rate_strong' in df.columns:
            df.plot(x='train_episode', y='eval_win_rate_strong', ax=ax2)
            ax2.set_title('Win Rate Strong')

        if 'eval_win_rate_self' in df.columns:
            df.plot(x='train_episode', y='eval_win_rate_self', ax=ax3)
            ax3.set_title('Win Rate Self')
        
        df.plot(x='train_episode', y='eval_reward', ax=ax4)
        ax4.set_title('Reward')
        
        plt.tight_layout()
        plt.savefig(self.metric_csv_file.replace('.csv', '_metrics.png'))
        plt.close()



