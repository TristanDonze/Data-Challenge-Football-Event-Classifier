import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
import torch
from torch.utils.data import TensorDataset, DataLoader


class ColorHue:
    # matplotlib.colors how to do a colormap which when it's 0 put red and when it's 1 put blue
    LABELS_HUE = LinearSegmentedColormap.from_list('labels_hue', ['blue' ,'violet', 'red'])


class Dataset:
    TRAIN = None
    TEST = None
    VALIDATION = None

    PROPORTION_TRAIN = 0.8

    PLOT_SIZE = (10, 8)

    def __init__(self, path: str, files: list[str], labeled=True):
        self.path = path
        self.files = files

        self.labeled = labeled
        self.labels_columns = ['EventType']

        self.brut_data = self.get_load_data()
        self.cleared_data = self.get_clear_data()
        self.enrich_cleared_data = self.get_enrich_data(self.cleared_data)
        self.enrich_brut_data = None

        self.num_features = -1
        self.setup_data: dict[str, DataLoader] | None = None
        self.setup_info = {"source": "", "time_step": -1, "batch_size": -1, "shuffle": False}

    def __repr__(self):
        txt = f"\n | Dataset(path='{self.path}', files={self.files}, labeled={self.labeled})"
        txt += f"\n | {'No setup' if self.setup_data is None else f'Setup: {self.setup_info}'}"
        return txt + '\n |______'

    def __iter__(self):
        """
        Iterate over the enriched cleared data on (file_name, DataFrame) pairs
        """

        if self.setup_data is None:
            raise ValueError("You should call the setup method before iterating over the dataset")
        return iter(self.setup_data.items())

    def _tweet_is_clean(self, tweet: str) -> bool:
        """
        Check if the tweet is clean or not

        ### Arguments:
            tweet: str
            The tweet to check

        ### Returns:
            bool
            True if the tweet is clean, False otherwise
        """
        if tweet.startswith("RT"):
            return True
        if "http" in tweet:
            return True
        if "@" in tweet:
            return True
        return False

    def _preprocess_temporal_data(self, data: dict, batch_size: int, timestep: int, shuffle: bool) -> dict:
        """
        Preprocess the temporal data

        ### Arguments:
            data: dict
            The data to preprocess

            batch_size: int
            The size of the batch

            shuffle: bool
            Shuffle the data or not

        ### Returns:
            data: dict
            The preprocessed data
        """
        for file_name, df in data.items():
            X = df[df.columns.difference(self.labels_columns)].values

            if self.labeled:
                y = df[self.labels_columns].values
            else:
                y = np.zeros(len(X))

            data[file_name] = self._prepocess_temporal_datum(X, y, batch_size, timestep, shuffle)

        return data

    def _prepocess_temporal_datum(self, X: np.ndarray, y: np.ndarray, batch_size: int, timestep: int, shuffle: bool) -> DataLoader:
        """
        Pré-traite les données temporelles.
        """
        half_step = timestep // 2
        decalage = timestep % 2 == 0

        X_seq = []
        y_seq = []

        # Ajouter les premiers half_steps éléments complétés avec des zéros
        for i in range(half_step):
            zeros_padding = np.zeros((half_step - i, self.num_features))
            X_slice = X[:i + half_step + 1 - decalage]
            X_seq.append(np.concatenate([zeros_padding, X_slice]))
            y_seq.append(y[i])

        # Ajouter les éléments centraux
        for i in range(half_step, len(X) - half_step + decalage):
            X_seq.append(X[i - half_step : i + half_step + 1 - decalage])
            y_seq.append(y[i])

        # Ajouter les derniers half_step éléments complétés avec des zéros
        for index in range(len(X) - half_step + decalage, len(X)):
            padding_right = index + half_step + 1 - len(X)
            if decalage:
                padding_right -= 1
            zeros_padding = np.zeros((padding_right, self.num_features))
            X_seq.append(np.concatenate([X[index - half_step:], zeros_padding]))
            y_seq.append(y[index])

        X_seq = np.array(X_seq)  # Forme : (num_samples, timesteps, num_features)
        y_seq = np.array(y_seq)

        # Convertir en tenseurs PyTorch
        X_seq = torch.from_numpy(X_seq).float()
        y_seq = torch.from_numpy(y_seq).long() 

        dataset = TensorDataset(X_seq, y_seq)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader

    def setup(self, source: str, batch_size: int, timestep: int, shuffle: bool = True):
        """
        Setup the dataset for training

        ### Arguments:
            source: str
            The source of the data
            Should be in ['brut', 'cleared', 'brut_enrich', 'cleared_enrich']

            batch_size: int
            The size of the batch

            shuffle: bool
            Shuffle the data or not
        """
        if source not in ['brut', 'cleared', 'brut_enrich', 'cleared_enrich']:
            raise ValueError(f"source should be in ['brut', 'cleared', 'brut_enrich', 'cleared_enrich'], got {source}")

        if source == 'brut':
            data = self.brut_data
        elif source == 'cleared':
            data = self.cleared_data
        elif source == 'brut_enrich':
            if self.enrich_brut_data is None:
                self.enrich_brut_data = self.get_enrich_data(self.brut_data)
            data = self.enrich_brut_data
        elif source == 'cleared_enrich':
            if self.enrich_cleared_data is None:
                self.enrich_cleared_data = self.get_enrich_data(self.cleared_data)
            data = self.enrich_cleared_data

        self.num_features = next(iter(data.values())).shape[1] - len(self.labels_columns)
        self.setup_data = self._preprocess_temporal_data(data, batch_size, timestep, shuffle)
        self.setup_info = {"source": source, "time_step": timestep, "batch_size": batch_size, "shuffle": shuffle}
    
    def get_timestep(self) -> int:
        return self.setup_info["time_step"]
    
    def set_timestep(self, timestep: int):
        if self.setup_data is None:
            raise ValueError("You should call the setup method before setting the timestep")
        self.setup(self.setup_info["source"], self.setup_info["batch_size"], timestep, self.setup_info["shuffle"])

    def get_load_data(self) -> dict:
        """
        Load the data from the files with Pandas

        ### Returns:
            data: dict `{file_name: DataFrame}`
            A dictionary with the name of the file as key and the DataFrame as value
        """
        data = {}
        for file in self.files:
            file_name = file.replace('.csv', '').lower()
            data[file_name] = pd.read_csv(os.path.join(self.path, file))

        return data

    def get_clear_data(self, reverse=False) -> dict:
        """
            Clear the data by removing the rows where the `self.tweet_is_clean(Tweet)` is False
            Can be reversed to work on the rows where the `self.tweet_is_clean(Tweet)` is False

            ### Returns:
                data: dict `{file_name: DataFrame}`
                A dictionary with the name of the file as key and the DataFrame as value
        """

        if self.brut_data is None:
            self.brut_data = self.get_load_data()

        data = {}

        if reverse:
            for file_name, df in self.brut_data.items():
                data[file_name] = df[df['Tweet'].apply(self._tweet_is_clean)]
        else:
            for file_name, df in self.brut_data.items():
                data[file_name] = df[~df['Tweet'].apply(self._tweet_is_clean)]

        return data

    def get_enrich_data(self, data: dict) -> dict:
        """
        Enrichit les données avec des caractéristiques calculées :
            - Nombre de Tweets pour chaque PeriodID
            - Longueur moyenne des Tweets pour chaque PeriodID
        """

        for _, df in data.items():
            if self.labeled:
                agg_data = df.groupby('PeriodID').agg(
                    NbTweets=('Tweet', 'count'),
                    AvgLength=('Tweet', lambda x: x.str.len().mean()),
                    EventType=('EventType', 'first'),
                    PeriodId=('PeriodID', 'first'),
                ).reset_index()
            
            else:
                agg_data = df.groupby('PeriodID').agg(
                    NbTweets=('Tweet', 'count'),
                    AvgLength=('Tweet', lambda x: x.str.len().mean()),
                    PeriodId=('PeriodID', 'first'),
                    MatchID=('MatchID', 'first'),
                ).reset_index()

            data[_] = agg_data

        return data

    def plot(self, source: str, agg_func: str, x: str, y: str, z: str | None = None, color: str | None = None,
             title: str = "", xlabel: str = "", ylabel: str = "", zlabel: str = "",
             color_hue: Colormap = None, show_hue_legend: bool = True, plot_style: dict = None, is_line_chart: bool=True):
        """
        Plot the data

        ### Arguments:
            source: str
            The name of data source

            agg_func: str
            The aggregation function to use to plot the data of the different matches
            Should be in: ['mean', 'subplots', 'stacked']

            x: str
            The column to use as x-axis

            y: str
            The column to use as y-axis

            z: str|None
            The column to use as z-axis

            color: str|None
            The column to use as color

            title: str
            The title of the plot

            xlabel: str
            The label of the x-axis

            ylabel: str
            The label of the y-axis

            zlabel: str
            The label of the z-axis

            color_hue: Colormap
            The colormap to use for coloring

            show_hue_legend: bool
            Show the hue legend

            plot_style: dict
            The style of the plot
            { 'linewidth': ..., ...} matplotlib style

            is_line_chart: bool
            True if the plot is a line chart, False otherwise (scatter plot)
        """

        if source not in ['brut', 'cleared', 'brut_enrich', 'cleared_enrich']:
            raise ValueError(f"source should be in ['brut', 'cleared', 'brut_enrich', 'cleared_enrich'], got {source}")
        if agg_func not in ['mean', 'subplots', 'stacked']:
            raise ValueError(f"agg_func should be in ['mean', 'subplots', 'stacked'], got {agg_func}")

        if source == 'brut':
            data = self.brut_data
        elif source == 'cleared':
            data = self.cleared_data
        elif source == 'brut_enrich':
            if self.enrich_brut_data is None:
                self.enrich_brut_data = self.get_enrich_data(self.brut_data)
            data = self.enrich_brut_data
        elif source == 'cleared_enrich':
            if self.enrich_cleared_data is None:
                self.enrich_cleared_data = self.get_enrich_data(self.cleared_data)
            data = self.enrich_cleared_data

        if (color_hue is not None or color is not None) and is_line_chart:
            print("\033[93mWARNING: We cannot use a colormap for line charts, the color_hue will be ignored.\033[0m")
            color_hue = None
            color = None

        if agg_func == 'mean':
            df = pd.concat(data.values())
            # Sélectionner uniquement les colonnes numériques pour éviter les erreurs lors du calcul de la moyenne
            df_numeric = df.select_dtypes(include=[np.number])
            df_mean = df_numeric.groupby('PeriodID').mean().reset_index()
            # Si z est spécifié, il doit être une colonne numérique valide
            if z is not None and z not in df_mean.columns:
                raise ValueError(f"La colonne z '{z}' n'existe pas dans les données agrégées.")
            self._plot(df_mean, '', x, y, z, color, xlabel, ylabel, zlabel, color_hue, show_hue_legend, ax=None, plot_style=plot_style, is_line_chart=is_line_chart)
            plt.title(title)

        elif agg_func == 'subplots':
            # Définir le nombre de lignes et de colonnes, on doit être sur que tous les graphiques rentrent
            nb_elmt_per_row = 3
            nb_rows = len(data) // nb_elmt_per_row + (1 if len(data) % nb_elmt_per_row else 0)
            nb_cols = nb_elmt_per_row

            if z is not None: fig, axs = plt.subplots(nb_rows, nb_cols, subplot_kw={'projection': '3d'}, figsize=(5 * nb_cols, 4 * nb_rows))
            else: fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(5 * nb_cols, 4 * nb_rows))

            fig.suptitle(title)
            axs = axs.flatten()  # Aplatir l'array d'axes pour itération facile

            for idx, (file_name, df) in enumerate(data.items()):
                self._plot(df, file_name, x, y, z, color, xlabel, ylabel, zlabel, color_hue, show_hue_legend, ax=axs[idx], plot_style=plot_style, is_line_chart=is_line_chart)

            # Supprimer les axes inutilisés
            for i in range(len(data), len(axs)):
                fig.delaxes(axs[i])

        elif agg_func == 'stacked':
            # L'idée est de tracer une courbe par fichier
            if z is not None:
                fig = plt.figure(figsize=Dataset.PLOT_SIZE)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=Dataset.PLOT_SIZE)

            fig.suptitle(title)
            for idx, (file_name, df) in enumerate(data.items()):
                self._plot(df, file_name, x, y, z, color, xlabel, ylabel, zlabel, color_hue, show_hue_legend and idx == 0, ax=ax, plot_style=plot_style, is_line_chart=is_line_chart)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuster l'espacement pour le titre principal
        plt.show()

    def _plot(self, df: pd.DataFrame, file_name: str, x: str, y: str, z: str | None, color: str | None,
              xlabel: str, ylabel: str, zlabel: str, color_hue: Colormap, show_hue_legend: bool, ax=None, plot_style: dict = None, is_line_chart=True):
        """
        Plot a single DataFrame on the given axes

        ### Arguments:
            df: pd.DataFrame
            file_name: str
            x: str
            y: str
            z: str|None
            color: str|None
            xlabel: str
            ylabel: str
            zlabel: str
            color_hue: Colormap
            ax: matplotlib.axes.Axes|mpl_toolkits.mplot3d.Axes3D|None
        """
        if ax is None:
            if z is not None:
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            else:
                fig, ax = plt.subplots()

        if z is None:
            if color is None:
                if is_line_chart: to_draw = ax.plot(df[x], df[y], **plot_style)
                else: to_draw = ax.scatter(df[x], df[y], **plot_style)
            elif color_hue is None:
                if is_line_chart: to_draw = ax.plot(df[x], df[y], c=df[color], **plot_style)
                else: to_draw = ax.scatter(df[x], df[y], c=df[color], **plot_style)
            else:
                if is_line_chart: to_draw = ax.plot(df[x], df[y], c=df[color], cmap=color_hue, **plot_style)
                else: to_draw = ax.scatter(df[x], df[y], c=df[color], cmap=color_hue, **plot_style)
        else:
            # 3D plot
            if color is None:
                if is_line_chart: to_draw = ax.plot(df[x], df[y], df[z], **plot_style)
                else: to_draw = ax.scatter(df[x], df[y], df[z], **plot_style)
            elif color_hue is None:
                if is_line_chart: to_draw = ax.plot(df[x], df[y], df[z], c=df[color], **plot_style)
                else: to_draw = ax.scatter(df[x], df[y], df[z], c=df[color], **plot_style)
            else:
                
                if is_line_chart: to_draw = ax.scatter(df[x], df[y], df[z], c=df[color], cmap=color_hue, **plot_style)
                else: to_draw = ax.scatter(df[x], df[y], df[z], c=df[color], cmap=color_hue, **plot_style)

        # Remarquez que 'title' n'est pas passé comme argument, donc utilisez 'file_name' uniquement
        ax.set_title(file_name if file_name else "")
        if z is not None:
            ax.set_xlabel(xlabel if xlabel else x)
            ax.set_ylabel(ylabel if ylabel else y)
            ax.set_zlabel(zlabel if zlabel else z)
        else:
            ax.set_xlabel(xlabel if xlabel else x)
            ax.set_ylabel(ylabel if ylabel else y)

        if color is not None and show_hue_legend:
            cbar = plt.colorbar(to_draw, ax=ax)
            cbar.set_label(color)
        ax.grid(True)
        # ax.legend([file_name])

    @staticmethod
    def init(path, labeled_files=None, unlabeled_files=None):
        if labeled_files is None:
            labeled_files = os.listdir(f'{path}train_tweets')
        if unlabeled_files is None:
            unlabeled_files = os.listdir(f'{path}eval_tweets')

        train_files = [file for file in labeled_files if file.endswith('.csv')]
        eval_files = [file for file in unlabeled_files if file.endswith('.csv')]

        nb_train = int(len(train_files) * Dataset.PROPORTION_TRAIN)
        train_files, test_files = train_files[:nb_train], train_files[nb_train:]

        Dataset.TRAIN = Dataset(path=f'{path}train_tweets', files=train_files)
        Dataset.TEST = Dataset(path=f'{path}train_tweets', files=test_files)
        Dataset.VALIDATION = Dataset(path=f'{path}eval_tweets', files=eval_files, labeled=False)


if __name__ == '__main__':
    # Initialiser les datasets
    Dataset.init('./CaseStudy/data/')

    # Exemple d'appel de la méthode plot
    Dataset.TEST.plot(
        source='cleared_enrich',
        agg_func='subplots',  # in 'subplots', 'mean', 'stacked'
        x='PeriodID',
        y='PeriodID',
        z='PeriodID',
        color='EventType',
        title='Graphique de Moyenne',
        color_hue=ColorHue.LABELS_HUE,
        show_hue_legend=False,
        plot_style={'linewidth': 1},
        is_line_chart=False
    )