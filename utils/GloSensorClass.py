import pandas as pd 
import numpy as np 
import sys 
import os 
import copy
from csv import reader
import csv 
import plotly.express as px
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class GlosensorData():
    def __init__(self, path: str):
        self.file_dir = path
    

    def ReadTableData(self):
        '''
        ReadTableData combines data from .csv files exported from the MARS program's table view.
        
        Parameters:
            - file_dir (str): Path to the directory where the files are located.
            (Note that file names should be in the format of data_initial_platenumber_blankoptional,
            e.g., 12241994_JL_1_Blank)
        
        Returns:
            dict: A dictionary containing combined dataframes for each plate, where the plate name
                is derived from the file name.
        
        Example Usage:
            glo = GlosensorData(path)
            data_dict = glo.ReadTableData('/path/to/data/files/')
        '''
        files = self.getFileNames(self.file_dir)
        combine_data = {}
        for file in files:
            data = pd.DataFrame()
            
            if 'blank' in file.lower():
                plate = f"Plate_{file.split('.')[0].split('_')[2]}_blank" #Updates the plate # based on the 3rd index of '_'
            else:
                plate = f"Plate_{file.split('.')[0].split('_')[2]}" #Updates the plate # based on the 3rd index of '_'

            with open(os.path.join(self.file_dir + file), 'r') as file:
                reader = csv.reader(file)
                
                for line in reader:
                    try: 
                        # Reads line if the first element starts with row of plate, and fits plate format length of 'AXX'
                        if (line[0][0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']) & \
                            (len(line[0]) == 3):
                            row_data = pd.DataFrame([filter(None, line)])
                            data = pd.concat([data,row_data])
                    except: 
                        continue
            combine_data[plate] = data
            print(f'Read {plate} ')
                
        # return combine_data 
        all_plate_data = pd.DataFrame()
        for k in np.unique([k.split('_')[1] for k in combine_data.keys()]):
            blank_data = combine_data[f'Plate_{k}_blank']
            blank_data.columns = ['Position', 'sample', 'Time_0']

            plate_data = combine_data[f'Plate_{k}']
            plate_data.columns = ['Position', 'sample'] + [f'Time_{i}'for i in range(1,len(plate_data.columns)-1)]
            
            # Delete sample column 
            plate_data = plate_data.drop('sample', axis = 1)
            
            # insert blank Time_0 into plate_data 
            plate_data.insert(1, 'Time_0', blank_data['Time_0'])
            plate_data.insert(1, 'Plate', k.astype(int))
            
            all_plate_data = pd.concat([all_plate_data, plate_data])
        
        # Convert Position into Rows and Columns identifiers  
        all_plate_data.insert(1, 'Row', all_plate_data['Position'].str.split('(\d+)', expand=True)[0]) 
        all_plate_data.insert(1, 'Column', all_plate_data['Position'].str.split('(\d+)', expand=True)[1]) 
        all_plate_data['Column'] = all_plate_data['Column'].astype(int)
        
        # Since values from excel sheets are read as str. Convert values to int 
        self.time_col = [col for col in all_plate_data.columns if col.startswith('Time_')]
        all_plate_data[self.time_col] = all_plate_data[self.time_col].astype(int)
        
        # instantiate in class
        self.data = all_plate_data 
        # return all_plate_data 
    
    def ReadPlates(self, file_dir):
    
        """
        DEPRECATED, preferably save data as Table_view and use ReadTableData() instead.
        file_dir is the directory which the csv data is stored at 

        Returns the dataframe holding the 96-well data
        """    
        
        files = self.getFileNames(file_dir)
        # Using open to read csv files line by line. 
        self.data = pd.DataFrame(columns=['Position','Row','Column', 'Time', 'value','Plate'])
        for file in files:
            interval = 0 
            plate = file.split('.')[0].split('_')[2] #Updates the plate # based on the 3rd index of '_'
            with open(file_dir + file) as file: 
                file_reader = reader(file)
                for line in file_reader:
            #         If first element of the line contains Interval, save the interval time 
                    if not len(line) == 0:
                        if 'Cycle ' in line[0]:
                            interval = line[0].split(" ")[1]
            #             If first element of the line is in ['A', 'B'....] the plate rows save the row values
                        if line[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']: 
                            row_data = pd.DataFrame(columns=self.data.columns)
                            for i in range(1,len(line)):
                                if not line[i] == '': #Skip if the cell has no record
                                    row_data = pd.concat([row_data, pd.DataFrame([[line[0]+str(i),
                                                                            line[0], 
                                                                            i,
                                                                            interval,
                                                                            float(line[i]), 
                                                                            int(plate)]],
                                                                            columns=self.data.columns)])
                            self.data = pd.concat([self.data,row_data])
        # Adds numerical index to dataframe
        self.data.index = range(len(self.data))
    
    def getFileNames(self, file_dir):
        """
        Defines which directory to read in csv. 
        change directory_name to file directory 
        Files name should be in format of.. 
        DATE_INITIAL_PLATE#.csv
        """
        files = []
        for i in os.listdir(file_dir):   
            try:
                if i.split('.')[1] in ['csv', 'CSV']:
                    files.append(i)
            except:
                print(i + ' Skipped')
                continue    
        
        files.sort()
        return files
    
    def normalize_by_t(self,
                       data = None,
                       by_time = 0,
                       how = 'divide'  
                       ):
        '''
        Normalize data by a specific time point (Time_X) in a given DataFrame.

        Parameters:
            - data (DataFrame): The input data to be normalized.
            - by_time (int): The Time_X value to use as the reference for normalization.
            - how (str): Specifies the normalization method. Options are 'divide' (default) or 'subtract'.

        Returns:
            DataFrame: A DataFrame with the data normalized by the specified time point.

        Example Usage:
            glo = GlosensorData(path)
            t_norm_data = glo.normalize_by_t(data=my_data, by_time=5, how='divide')
        '''
        
        assert data is not None, "Please give input data."
        
        t_norm_data = data.copy()
        
        if how == 'divide': 
            t_norm_data[self.time_col] = t_norm_data[self.time_col].div(t_norm_data[f'Time_{by_time}'], axis=0)
        elif how == 'subtract':
            t_norm_data[self.time_col] = t_norm_data[self.time_col].subtract(t_norm_data[f'Time_{by_time}'], axis=0)
        else: 
            print("Please specify 'how' as one of the options ['divide', 'subtract']")
        
        print(f'data normalized by Time_{by_time} with {how}')
        return t_norm_data
    
    
    def normalize_by_pCI(self, 
                         data = None,
                         group_col = ['Plate', 'Odor', 'Odor_conc']
                         ):
        '''
        Normalize data by the Plate-Control Index (pCI) within specified groups.

        Parameters:
            - data (DataFrame): The input data to be normalized.
            - group_col (list): A list of columns used to group the data for pCI calculation.

        Returns:
            DataFrame: A DataFrame with the data normalized by the pCI values within the specified groups.

        Example Usage:
            glo = GlosensorData(path)
            pci_norm_data = glo.normalize_by_pCI(data=my_data, group_col=['Plate', 'Odor', 'Odor_conc'])
        '''
        
        assert data is not None, "Please give input data."
        
        pci_norm_data = data.copy()

        control_df = pci_norm_data[pci_norm_data['Group'].str.contains('pCI')]
        control_grouped = control_df.groupby(group_col)[self.time_col].mean().reset_index()
        pci_norm_data = pci_norm_data.merge(control_grouped, on=group_col, suffixes=('', '_pCI'))

        for t_col in self.time_col: 
            pci_norm_data[t_col] = pci_norm_data[t_col] / pci_norm_data[f'{t_col}_pCI']

        pci_norm_data.drop(columns=[f'{t_col}_pCI' for t_col in self.time_col], inplace=True)

        # Print statement for if there are plates without pCI hence dropped. 
        plates_without_pci = list(set(data.Plate.unique()).difference(set(control_df.Plate.unique())))
        if len(plates_without_pci) > 0:
            print(f'Plates {plates_without_pci} does not contain pCI.\nRemoved from data.')
        
        print('data normalized by pCI')
        return pci_norm_data
    
    def double_normalize(self, 
                         data = None, 
                         by_time = 0, 
                         how = 'divide', 
                         group_col = ['Plate', 'Odor', 'Odor_conc']):
        '''
        Perform double normalization on data by first normalizing by time (Time_X) and then by pCI within specified groups.

        Parameters:
            - data (DataFrame): The input data to be double normalized.
            - by_time (int): The Time_X value to use as the reference for the first normalization step.
            - how (str): Specifies the first normalization method. Options are 'divide' (default) or 'subtract'.
            - group_col (list): A list of columns used to group the data for pCI calculation in the second normalization step.

        Returns:
            DataFrame: A DataFrame with the data double normalized by time and pCI.

        Example Usage:
            glo = GlosensorData(path)
            double_norm_data = reader.double_normalize(data=my_data, by_time=5, how='divide', group_col=['Plate', 'Odor', 'Odor_conc'])
        '''
        
        assert data is not None, "Please give input data."

        raw_data = data.copy()
        
        t_norm_data = self.normalize_by_t(raw_data, 
                                          by_time, 
                                          how)
        double_norm_data = self.normalize_by_pCI(t_norm_data, 
                                                 group_col)
        
        return double_norm_data
    

    def get_melted_AUC_slope_data(self, 
                           group_col: list, 
                           slope_AUC_time = [0,5], 
                           data = None, 
                           AUC_method = 'trapz'):
        '''
        Extract and analyze data in a melted format including mean, standard deviation,
        slope, and area under the curve (AUC) values for specified groups.

        Parameters:
            - group_col (list): A list of columns used for grouping data.
            - slope_AUC_time (list): Time range to calculate slope AUC, specified as [start_time, end_time].
            - data (DataFrame): The input data in a melted format.

        Returns:
            - melted_data (DataFrame): The melted data with calculated mean, standard deviation, and slopes.
            - AUC_data (DataFrame): A DataFrame with calculated raw AUC values and statistics.
            - slope_data (DataFrame): A DataFrame with calculated slope AUC values and statistics.

        Example Usage:
            glo = GlosensorData(path)
            melted, AUC, slopes = reader.get_melted_AUC_slope_data(
                group_col=['Plate', 'Odor', 'Odor_conc', 'Replicate'],
                slope_AUC_time=[0, 5],
                data=my_melted_data
            )
        '''
        assert data is not None, "Please give input data."
        
        # Define replicate columns. Replicate for rows and values that share the same group_col 
        data['Replicate'] = data.groupby(group_col).cumcount() + 1 
        group_col = group_col + ['Replicate']
        
        
        # Get a list of colnames for Time 
        time_col = [col for col in data.columns if col.startswith('Time_')]
        
        # Create melted_data for calculating mean, std etc. 
        melted_data = pd.melt(data, 
                                   id_vars=group_col+['Plate'], 
                                   value_vars=time_col, var_name='Time', value_name='value')
        melted_data['Time'] = melted_data['Time'].str.replace('Time_', '').astype(int)

        # Calculate mean and std of replicates
        melted_data['mean'] = melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['value'].transform('mean')
        melted_data['std'] = melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['value'].transform('std')
        # Calcuate the slope 
        melted_data['slope'] = ((melted_data['value'] - 
                                            melted_data.groupby(group_col)['value'].shift(1)) / 100).replace(np.nan, 0)
        # Calculate slope_std by taking std across replicates 
        melted_data['slope_mean'] = melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['slope'].transform('mean')
        melted_data['slope_std'] = melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['slope'].transform('std')

        # Calculate raw AUC values
        if AUC_method == 'trapz': 
            AUC_data = melted_data.groupby(group_col, as_index=False).apply(lambda group: group.assign(AUC = np.trapz(group['value'], group['Time']))).reset_index(drop=True)
        else: 
            AUC_data = melted_data.groupby(group_col, as_index=False).apply(lambda group: group.assign(AUC = np.sum(group['value']))).reset_index(drop=True)
        AUC_data['AUC_mean'] = AUC_data.groupby([i for i in group_col if i != 'Replicate'])['AUC'].transform('mean')
        AUC_data['AUC_std'] = AUC_data.groupby([i for i in group_col if i != 'Replicate'])['AUC'].transform('std')
        # AUC_data = AUC_data.drop_duplicates(subset = [i for i in group_col if i != 'Replicate']).reset_index(drop = True)

        # Calculate Slope AUC values with defined number of slopes to use
        slope_data = melted_data[melted_data['Time'].between(slope_AUC_time[0], slope_AUC_time[1])]
        if AUC_method == 'trapz': 
            slope_data = slope_data.groupby(group_col, as_index=False).apply(lambda group: group.assign(slope_AUC = np.trapz(group['slope'], group['Time']))).reset_index(drop=True)
        else: 
            slope_data = slope_data.groupby(group_col, as_index=False).apply(lambda group: group.assign(slope_AUC = np.sum(group['slope']))).reset_index(drop=True)
        slope_data['slope_AUC_mean'] = slope_data.groupby([i for i in group_col if i != 'Replicate'])['slope_AUC'].transform('mean')
        slope_data['slope_AUC_std'] = slope_data.groupby([i for i in group_col if i != 'Replicate'])['slope_AUC'].transform('std')
        # slope_data = slope_data.drop_duplicates(subset = [i for i in group_col if i != 'Replicate']).reset_index(drop = True)

        return melted_data, AUC_data, slope_data 

    def view_96well_values(self, 
                          plate = None, 
                          time = 0, 
                          cmap = 'RdBu'):
        '''
        Generate a heatmap-like visualization of 96-well plate values at a specific time point.

        Parameters:
            - plate (str): The plate identifier.
            - time (int): The time point to display values for.
            - cmap (str): The colormap to use for the heatmap (default is 'RdBu').

        Returns:
            pandas.io.formats.style.Styler: A stylized DataFrame representing the 96-well plate values.

        Example Usage:
            plate_view = reader.view_96well_values(plate='Plate_1', time=0, cmap='RdBu')
        '''
    
        assert self.data is not None, "Please read table/plate data before you can plot_plate_view"

        plot_data = self.data.copy()
        plot_data = plot_data[plot_data.Plate == plate]
        plot_data = plot_data[list(plot_data.columns[0:4]) +
                              list(plot_data.columns[plot_data.columns == 'Time_' + str(time)])]
        
        # Create a new dataframe in the 96-well format
        formatted_df = pd.pivot_table(plot_data, 
                                      values='Time_'+str(time), 
                                      index='Row', 
                                      columns='Column')

        # Reorder the columns to match 'col1-12' left to right
        formatted_df = formatted_df[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

        # Reorder the rows to match 'A-H' top to bottom
        formatted_df = formatted_df.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

        print(f"Plate {plate} at Time_{time} values.")
        return formatted_df.style.background_gradient(cmap=cmap)
        # return formatted_df
    
        

    def view_96well_curve(self, 
                        x_by = 'Time', 
                        y_by = 'value',
                        facet_col_by = 'Column',
                        facet_row_by = 'Row',
                        facet_col_spacing = 0.03, 
                        facet_row_spacing = 0.03, 
                        n_facet_col_wrap = 12,
                        font_size = 10
                        ):
        '''
        Create facetted line plots for 96-well plate data to visualize curve profiles.

        Parameters:
            - x_by (str): The column to use for the x-axis.
            - y_by (str): The column to use for the y-axis.
            - facet_col_by (str): The column to use for faceting the plot into columns.
            - facet_row_by (str): The column to use for faceting the plot into rows.
            - facet_col_spacing (float): Spacing between facet columns (default is 0.03).
            - facet_row_spacing (float): Spacing between facet rows (default is 0.03).
            - n_facet_col_wrap (int): The maximum number of columns for faceting (default is 12).
            - font_size (int): Font size for the plot (default is 10).

        Returns:
            list: A list of Plotly figures representing facetted line plots for the 96-well plate data.

        Example Usage:
            reader = DataReader()
            curve_plots = reader.view_96well_curve(
                x_by='Time', y_by='value', facet_col_by='Column',
                facet_row_by='Row', facet_col_spacing=0.03,
                facet_row_spacing=0.03, n_facet_col_wrap=12, font_size=10
            )
        '''
        assert self.data is not None, "Please read table/plate data before you can plot_plate_view"
        
        melted_data = pd.melt(self.data, 
                     id_vars=['Plate', 'Column', 'Row'], 
                     value_vars=self.time_col, var_name='Time', value_name='value')
        
        fig_list = []
        for p in melted_data['Plate'].unique():
            plot_data = melted_data.copy()
            plot_data = plot_data[plot_data['Plate'] == p]
            
            fig = px.line(plot_data, 
                            x=x_by,
                            y=y_by,
                            facet_col=facet_col_by,
                            facet_row=facet_row_by,
                            facet_col_spacing=facet_col_spacing,
                            facet_row_spacing=facet_row_spacing,
                            facet_col_wrap=n_facet_col_wrap,
                            )

            fig.update_traces(marker=dict(size=8),
                                selector=dict(mode='markers'))

            # This lines disables auto-sizing of the y axis when toggling data
            fig.update_layout(yaxis_range =[(plot_data['value'].min()*0.9), 
                                            (plot_data['value'].max()*1.1)])

            # fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

            # hide subplot y-axis titles and x-axis titles
            for axis in fig.layout:
                if type(fig.layout[axis]) == go.layout.YAxis:
                    fig.layout[axis].title.text = ''
                if type(fig.layout[axis]) == go.layout.XAxis:
                    fig.layout[axis].title.text = ''
                    
            fig.update_layout(
                title="Plate_"+str(p),
                # xaxis_title="X Axis Title",
                # yaxis_title="Y Axis Title",
                font=dict(
                    size=font_size
                )
            )
                    
            fig_list.append(fig)

        return fig_list
        

    
    def plot_facet_line(self, df : pd.DataFrame,
                        color_by = 'Odor_conc',
                        facet_col_by = 'Group', 
                        facet_row_by = None, 
                        n_facet_col_wrap = 4,
                        x_by = "Time", 
                        y_by = "mean", 
                        error_y_by = 'std'
                        ):
        '''
        Create a facetted line plot using Plotly Express for visualizing data with multiple facets.

        Parameters:
            - df (pd.DataFrame): The input DataFrame containing the data to be plotted.
            - color_by (str): The column to use for color-coding lines in the plot.
            - facet_col_by (str): The column to use for faceting the plot into columns.
            - facet_row_by (str): The column to use for faceting the plot into rows (optional).
            - n_facet_col_wrap (int): The maximum number of columns for faceting (default is 4).
            - x_by (str): The column to use for the x-axis.
            - y_by (str): The column to use for the y-axis.
            - error_y_by (str): The column to use for error bars on the y-axis (default is 'std').

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure object representing the facetted line plot.

        Example Usage:
            line_plot = glo.plot_facet_line(
                df=my_data,
                color_by='Odor_conc',
                facet_col_by='Group',
                facet_row_by='Replicate',
                n_facet_col_wrap=4,
                x_by='Time',
                y_by='mean',
                error_y_by='std'
            )
        '''
        plot_data = df.copy()

        # facet_col_order = list(glo_data.melted_data.sort_values(['Group']).unique())
        legend_order = list(plot_data['Odor_conc'].astype(float)\
                            .sort_values(ascending=False).unique())

        # Convert Odor concentration to str if it isn't already 
        if not type(list(plot_data[color_by])[0]) == str:
            plot_data[color_by] = plot_data[color_by].astype(str)

        fig = px.line(plot_data, 
                        x=x_by, 
                        y=y_by,
                        error_y = error_y_by, 
                        color=color_by,
                        facet_col=facet_col_by,
                        facet_row=facet_row_by,
                        facet_col_wrap=n_facet_col_wrap,
                        category_orders={
                                        #   Y: facet_col_order,
                        color_by: legend_order}
                        )

        fig.update_traces(marker=dict(size=8),
                          selector=dict(mode='markers'))
        # This line makes facet y axis free (If Used, Comment the line below )
        # fig.update_yaxes(matches=None)

        # This lines disables auto-sizing of the y axis when toggling data
        fig.update_layout(yaxis_range =[(plot_data[y_by].min()*0.9), 
                                        (plot_data[y_by].max()*1.1)])

        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig 
    
    def plot_facet_bar(self, df : pd.DataFrame,
                       color_by = 'Odor_conc',
                       facet_col_by = 'Group', 
                       facet_row_by = None, 
                       n_facet_col_wrap = 4,
                       error_y_by = 'slope_std',
                       x_by = "Odor_conc", y_by = "slope_AUC"
                       ):
        '''
        Create a facetted bar plot using Plotly Express for visualizing data with multiple facets.

        Parameters:
            - df (pd.DataFrame): The input DataFrame containing the data to be plotted.
            - color_by (str): The column to use for color-coding bars in the plot.
            - facet_col_by (str): The column to use for faceting the plot into columns.
            - facet_row_by (str): The column to use for faceting the plot into rows (optional).
            - n_facet_col_wrap (int): The maximum number of columns for faceting (default is 4).
            - error_y_by (str): The column to use for error bars on the y-axis (default is 'slope_std').
            - x_by (str): The column to use for the x-axis.
            - y_by (str): The column to use for the y-axis.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure object representing the facetted bar plot.

        Example Usage:
            bar_plot = glo.plot_facet_bar(
                df=my_data,
                color_by='Odor_conc',
                facet_col_by='Group',
                facet_row_by='Replicate',
                n_facet_col_wrap=4,
                error_y_by='slope_std',
                x_by='Odor_conc',
                y_by='slope_AUC'
            )
        '''
        
        plot_data = df.copy()
        
        legend_order = list(plot_data['Odor_conc'].astype(float)\
                            .sort_values(ascending=False).unique())


        if facet_row_by:
            fig = px.bar(plot_data, 
                            x=x_by, 
                            y=y_by,
                            color=x_by,
                            error_y=error_y_by,
                            facet_row=facet_row_by,
                            facet_col=facet_col_by,
                            facet_col_wrap=n_facet_col_wrap,
                            category_orders={
                                #  Y: facet_col_order,
                                            "Odor_conc": legend_order}
                            )
        else: 
            fig = px.bar(plot_data, 
                            x=x_by, 
                            y=y_by,
                            color=color_by,
                            error_y=error_y_by,
                            facet_col=facet_col_by,
                            facet_col_wrap=n_facet_col_wrap,
                            category_orders={
                                            #   Y: facet_col_order,
                                            color_by: legend_order}
                            )

        # fig.update_yaxes(matches=None)

        x_order = list(plot_data[x_by].astype(float).sort_values().unique().astype(str))
        fig.update_xaxes(categoryorder='array', categoryarray= x_order)
        # fig.update_layout(showlegend = False)

        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


        # This lines disables auto-sizing of the y axis when toggling data
        fig.update_layout(yaxis_range =[(plot_data[y_by].min()*0.9), 
                                        (plot_data[y_by].max()*1.1)])


        return fig 
    
    def plot_Dose_curves(self, 
                         data_lists,
                         groupby_col = ['Group', 'Odor'], 
                        custom_color = None, 
                        custom_marker = None, 
                        figsize=(10,7),
                        point_size = 5, 
                        zero_conc = -20, 
                        log_conc_offset = -6, # -3 assuming the odor_conc enterred as uM. 
                        ignore_warning=True, 
                        plot_points = False, 
                        plot_std = True, 
                        std_capsize = 10, 
                        std_linewidth = 2, 
                        curve_width = 5, 
                        curve_alpha = 0.5, 
                        x_by = 'Odor_conc', 
                        y_by = 'AUC', 
                        sortlegend_by = 'AUC_mean', 
                        labelsize=20):

        """
        Plot sigmoidal dose-response curves for multiple datasets.

        Parameters:
            data_lists (DataFrame): A pandas DataFrame containing the data to plot.
            groupby_col (list, optional): A list of column names to group the data by.
                                        Default is ['Group', 'Odor'].
            custom_color (dict, optional): A dictionary mapping group names to custom colors.
                                        Default is None, in which case colors are generated automatically.

        Returns:
            matplotlib.figure.Figure: The generated figure containing the dose-response curves.
        """

        # Custom dose-response function (https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_dr_stim.htm)
        def dose_response(x, Bottom, Top, LogEC50):
            y = Bottom + (Top - Bottom) / (1 + 10**(LogEC50 - x))
            return y 

        def sigmoid(x, L, x0, k, b):
            y = L / (1 + np.exp(-k*(x-x0))) + b
            return y


        # Surpress RuntimeWarning
        if ignore_warning: 
            np.seterr(all='ignore')


        # Initialize the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, 
                                        figsize=figsize,
                                        gridspec_kw={'width_ratios': [1, 5]})
        if sortlegend_by: 
            data_lists = data_lists.sort_values(sortlegend_by, ascending=False).reset_index(drop=True)

        # Plot each curve with different colors
        if custom_color is None: 
            custom_color = continuous_colors(['_'.join(map(str, name)) for name, _ in data_lists.groupby(groupby_col, sort=False)],
                                        colormap = 'viridis_r')
        elif type(custom_color) == str: 
            color = custom_color
            custom_color = {}
            for i, _group in enumerate(['_'.join(map(str, name)) for name, _ in data_lists.groupby(groupby_col, sort=False)]): 
                custom_color[_group] = color
        else: 
            colors = custom_color
            custom_color = {}
            for i, _group in enumerate(['_'.join(map(str, name)) for name, _ in data_lists.groupby(groupby_col, sort=False)]): 
                custom_color[_group] = colors[i]
                
        if custom_marker is None: 
            custom_marker = distinct_markers(['_'.join(map(str, name)) for name, _ in data_lists.groupby(groupby_col, sort=False)])
        elif type(custom_marker) == str:
            marker = custom_marker
            custom_marker = {}
            for i, _group in enumerate(['_'.join(map(str, name)) for name, _ in data_lists.groupby(groupby_col, sort=False)]): 
                custom_marker[_group] = marker
        else: 
            marker = custom_marker
            custom_marker = {}
            for i, _group in enumerate(['_'.join(map(str, name)) for name, _ in data_lists.groupby(groupby_col, sort=False)]): 
                custom_marker[_group] = marker[i] 
                
        max_xdata = -np.inf
        min_xdata = np.inf
        stat_values = {}

        for i, (name, group) in enumerate(data_lists.groupby(groupby_col, sort=False)):
            name = '_'.join(map(str, name))
            
            group = group.drop_duplicates(y_by)
            xdata = group.Odor_conc
            xdata = [zero_conc if i == -np.inf else i for i in np.round(np.log10(xdata), 1)]
            ydata = group[y_by]
            x_unique = [zero_conc if i == -np.inf else i + log_conc_offset for i in np.round(np.log10(group.Odor_conc.unique()), 1)]
            y_mean = group.groupby(x_by, sort=False).apply(lambda x: x[y_by].mean())
            y_std = group.groupby(x_by, sort=False).apply(lambda x: x[y_by].std())

            # Plot data points
            if plot_points:
                xdata_plot = [zero_conc if i == zero_conc else i + log_conc_offset for i in xdata]
                ax1.plot(xdata_plot, ydata, 'o', 
                            color=custom_color[name])
                ax2.plot(xdata_plot, ydata, 'o', 
                            color=custom_color[name])
            if plot_std: 
                ax1.errorbar(x_unique, y_mean, y_std, 
                                linestyle='None', 
                                label = name, 
                                marker=custom_marker[name], 
                                markersize=point_size,
                                capsize=std_capsize, 
                                linewidth=std_linewidth, 
                                markeredgewidth=std_linewidth,
                                color = custom_color[name])
                ax2.errorbar(x_unique, y_mean, y_std, 
                                linestyle='None', 
                                label = name, 
                                marker=custom_marker[name], 
                                markersize=point_size, 
                                capsize=std_capsize,
                                markeredgewidth=std_linewidth,
                                linewidth=std_linewidth, 
                                color = custom_color[name])

            # Fit sigmoid curve
            p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
            try:
                popt, pcov = curve_fit(dose_response, xdata, ydata)
                curve_used = 'dose_response'
            except RuntimeError:
                try:
                    popt, pcov = curve_fit(sigmoid, xdata, ydata)
                    curve_used = 'sigmoid'
                except RuntimeError:
                    print("TRF method failed, trying 'lm' with increased max_nfev...")
                    popt, pcov = curve_fit(sigmoid, xdata, ydata, method='lm', max_nfev=10000)
                    curve_used = 'sigmoid'

            # Update min and max xdata for ax2
            max_xdata = max(max_xdata, max([x for x in xdata if x > zero_conc], default=-np.inf))
            min_xdata = min(min_xdata, min([x for x in xdata if x > zero_conc], default=np.inf))
            
            
            x_left = np.linspace(zero_conc-0.5, zero_conc+0.5)
            y_left =  dose_response(x_left, *popt) if curve_used == 'dose_response' else sigmoid(x_left, *popt) 
            x_right = np.linspace(min_xdata-0.5, max_xdata+0.5, 100)
            y_right = dose_response(x_right, *popt) if curve_used == 'dose_response' else sigmoid(x_right, *popt) 
            

            # Plot sigmoid curve
            ax1.plot(x_left, y_left, 
                    #  label=name, 
                    linewidth=curve_width, 
                    alpha = curve_alpha, 
                    color=custom_color[name])
            ax2.plot(x_right + log_conc_offset, y_right, 
                    #  label=name, 
                    linewidth=curve_width,
                    alpha = curve_alpha, 
                    color=custom_color[name])
            
            # Calculate EC50
            if curve_used == 'dose_response': 
                stat_values[name] = {}
                stat_values[name]['EC50'] = np.round(popt[2] + log_conc_offset, 2)
                stat_values[name]['Max'] = np.round(popt[1] + log_conc_offset, 2)
                
            else: 
                stat_values[name] = np.nan

        # Set x-axis limits and add vertical lines for breaks
        ax1.set_xlim(zero_conc-0.5, zero_conc+0.5)
        ax2.set_xlim(min_xdata-0.5 + log_conc_offset, max_xdata+0.5 + log_conc_offset)

        # Add labels and legend
        ax1.set_ylabel(y_by)
        ax2.set_xlabel('Odor Concentration (log10)')

        # Set ticks for the left subplot
        ax1.set_xticks([zero_conc])
        # ax1.set_xticks(['no_odor'])
        # ax1.set_xticklabels([zero_conc])

        # Remove y-axis ticks and labels from the second subplot
        ax1.tick_params(labelsize=labelsize)
        ax2.tick_params(left=False, labelleft=False, labelsize=labelsize)

        # Remove frames around the plots
        for i, ax in enumerate([ax1, ax2]):
            if i == 0:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else: 
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.8))

        # Show plot
        return fig, stat_values
    
    
def distinct_markers(label_list, num_markers=None, random_state=0):
    """
    Generate distinct markers for a list of labels.

    Parameters:
    label_list (list): A list of labels for which you want to generate distinct markers.
    num_markers (int): Number of distinct markers to generate. If None, it will use a default set of markers. Default is None.
    random_state (int): Seed for random number generation. Default is 0.

    Returns:
    dict: A dictionary where labels are keys and distinct markers are values.

    Example:
    >>> labels = ['A', 'B', 'C']
    >>> marker_mapping = distinct_markers(labels, num_markers=3)
    >>> print(marker_mapping)
    {'A': 'o', 'B': 's', 'C': 'D'}
    """
    random.seed(random_state)

    # Default set of markers
    default_markers = ['o', 's', 'D', '^', 'p', 'H', '*', '+', 'P', 'x', 'v', '<', '>', 'd']
    
    # Use default set of markers if number of markers is not specified
    if num_markers is None:
        num_markers = len(default_markers)

    # Sample distinct markers
    markers = random.sample(default_markers, min(num_markers, len(default_markers)))

    # Create marker dictionary
    marker_dict = {}
    for i, label in enumerate(label_list):
        marker_dict[label] = markers[i % len(markers)]

    return marker_dict
    
    
def distinct_colors(label_list, category=None, custom_color=None, random_state=0):
    """
    Generate distinct colors for a list of labels.

    Parameters:
    label_list (list): A list of labels for which you want to generate distinct colors.
    category (str): Category of distinct colors. Options are 'warm', 'floral', 'rainbow', or None for random. Default is None.

    Returns:
    dict: A dictionary where labels are keys and distinct colors (in hexadecimal format) are values.

    Example:
    >>> labels = ['A', 'B', 'C']
    >>> color_mapping = distinct_colors(labels, category='warm')
    >>> print(color_mapping)
    {'A': '#fabebe', 'B': '#ffd8b1', 'C': '#fffac8'}
    """
    random.seed(random_state)
    
    warm_colors = ['#fabebe', '#ffd8b1', '#fffac8', '#ffe119', '#ff7f00', '#e6194B']
    floral_colors = ['#bfef45', '#fabed4', '#aaffc3', '#ffd8b1', '#dcbeff', '#a9a9a9']
    rainbow_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
    pastel_colors = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', 
                     '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928', 
                     '#8DD3C7', '#BEBADA', '#FFED6F']
    
    color_dict = {}

    if custom_color is not None: 
        assert len(custom_color) >= len(label_list), "Provided label_list needs to be shorter than provided custom_color"
        for i, _label in enumerate(label_list): 
            color_dict[_label] = custom_color[i]
        return color_dict

    color_palette = None
    if category == 'warm':
        color_palette = random.sample(warm_colors, len(warm_colors))
    elif category == 'floral':
        color_palette = random.sample(floral_colors, len(floral_colors))
    elif category == 'rainbow':
        color_palette = random.sample(rainbow_colors, len(rainbow_colors))
    elif category == 'pastel': 
        color_palette = random.sample(pastel_colors, len(pastel_colors))
    else:
        color_palette = random.sample(warm_colors + floral_colors + rainbow_colors + pastel_colors, len(label_list))
    
    for i, label in enumerate(label_list):
        color_dict[label] = color_palette[i % len(color_palette)]
    
    return color_dict

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def continuous_colors(label_list, colormap='viridis', custom_color=None, orders=None):
    """
    Generate continuous colors for a list of labels.

    Parameters:
    label_list (list): A list of labels for which you want to generate continuous colors.
    colormap (str or matplotlib colormap, optional): The colormap to use for color scaling. Default is 'viridis'.
    custom_color (list, optional): A list of color tuples defining the custom colormap.
                                Default is None.
    orders (list, optional): A list defining the hierarchy of label_list. Default is None.

    Returns:
    dict: A dictionary where labels are keys and continuous colors (in hexadecimal format) are values.

    Example:
    >>> labels = ['A', 'B']
    >>> custom_color = [(0, '#DBE5EB'), (0.5, '#67879B'), (1, '#073763')]
    >>> color_mapping = continuous_colors(labels, custom_color=custom_color)
    >>> print(color_mapping)
    {'A': '#DBE5EB', 'B': '#073763'}
    """
    color_dict = {}

    # Choose colormap
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
    else:
        cmap = colormap

    # Generate custom colormap
    if custom_color is not None:
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_color)
    else:
        custom_cmap = None

    # Generate colors
    num_labels = len(label_list)
    
    for i, label in enumerate(label_list):
        if custom_cmap is not None:
            norm_color = i / (num_labels - 1)  # Normalize color index
            color = cm.colors.rgb2hex(custom_cmap(norm_color))
        else:
            color = cm.colors.rgb2hex(cmap(i / (num_labels - 1)))  # Normalize color index
        color_dict[label] = color

    # Reorder color_dict based on orders if provided
    if orders is not None:
        color_dict = {label: color_dict[label] for label in orders if label in color_dict}

    return color_dict

    
    # def plot_facet_scatter(self, df : pd.DataFrame,
    #                        color_by = 'Odor_conc',
    #                        facet_col_by = 'Group', 
    #                        facet_row_by = None, 
    #                        n_facet_col_wrap = 4,
    #                        x_by = "Time", y_by = "mean"
    #                        ):
    #     """
    #     DEPRECATED use plot_facet_line() instead
    #     """
    #     plot_data = df.copy()

    #     # facet_col_order = list(glo_data.melted_data.sort_values(['Group']).unique())
    #     legend_order = list(plot_data['Odor_conc'].astype(float)\
    #                         .sort_values(ascending=False).unique())

    #     # Convert Odor concentration to str if it isn't already 
    #     if not type(list(plot_data[color_by])[0]) == str:
    #         plot_data[color_by] = plot_data[color_by].astype(str)

    #     if facet_row_by: 
    #         fig = px.scatter(plot_data, 
    #                         x=x_by, 
    #                         y=y_by,
    #                         color=color_by,
    #                         facet_col=facet_col_by,
    #                         facet_row=facet_row_by,
    #                         facet_col_wrap=n_facet_col_wrap,
    #                         category_orders={
    #                                         #   Y: facet_col_order,
    #                                         color_by: legend_order}
    #                         )
    #     else: 
    #         fig = px.scatter(plot_data, 
    #                         x=x_by, 
    #                         y=y_by,
    #                         color=color_by,
    #                         facet_col=facet_col_by,
    #                         facet_col_wrap=n_facet_col_wrap,
    #                         category_orders={
    #                                         #   Y: facet_col_order,
    #                                         color_by: legend_order}
    #                         )
    #     fig.update_traces(marker=dict(size=8),
    #                       selector=dict(mode='markers'))
    #     # This line makes facet y axis free (If Used, Comment the line below )
    #     # fig.update_yaxes(matches=None)

    #     # This lines disables auto-sizing of the y axis when toggling data
    #     fig.update_layout(yaxis_range =[(plot_data[y_by].min()*0.9), 
    #                                     (plot_data[y_by].max()*1.1)])

    #     fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    #     fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    #     return fig 
    
    # def normalize_data(self, 
    #                    group_col: list, 
    #                    slope_AUC_time = [0,5]):
    #     """
    #     DEPRECATED
    #     """
        
    #     # Define replicate columns. Replicate for rows and values that share the same group_col 
    #     self.data['Replicate'] = self.data.groupby(group_col).cumcount() + 1 
    #     group_col = group_col + ['Replicate']
        
        
    #     # Get a list of colnames for Time 
    #     time_col = [col for col in self.data.columns if col.startswith('Time_')]
        
    #     # Create melted_data for calculating mean, std etc. 
    #     self.melted_data = pd.melt(self.data, 
    #                                id_vars=group_col+['Plate'], 
    #                                value_vars=time_col, var_name='Time', value_name='value')
    #     self.melted_data['Time'] = self.melted_data['Time'].str.replace('Time_', '').astype(int)

    #     # Calculate mean and std of replicates
    #     self.melted_data['mean'] = self.melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['value'].transform('mean')
    #     self.melted_data['std'] = self.melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['value'].transform('std')
    #     # Calcuate the slope 
    #     self.melted_data['slope'] = ((self.melted_data['value'] - 
    #                                         self.melted_data.groupby(group_col)['value'].shift(1)) / 100).replace(np.nan, 0)
    #     # Calculate slope_std by taking std across replicates 
    #     self.melted_data['slope_mean'] = self.melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['slope'].transform('mean')
    #     self.melted_data['slope_std'] = self.melted_data.groupby([i for i in group_col if i != 'Replicate'] + ['Time'])['slope'].transform('std')

    #     # Calculate raw AUC values
    #     self.AUC_data = self.melted_data.groupby(group_col, as_index=False).apply(lambda group: group.assign(AUC = np.trapz(group['value'], group['Time']))).reset_index(drop=True)
    #     self.AUC_data['AUC_mean'] = self.AUC_data.groupby([i for i in group_col if i != 'Replicate'])['AUC'].transform('mean')
    #     self.AUC_data['AUC_std'] = self.AUC_data.groupby([i for i in group_col if i != 'Replicate'])['AUC'].transform('std')
    #     # self.AUC_data = self.AUC_data.drop_duplicates(subset = [i for i in group_col if i != 'Replicate']).reset_index(drop = True)

    #     # Calculate Slope AUC values with defined number of slopes to use
    #     self.slope_data = self.melted_data[self.melted_data['Time'].between(slope_AUC_time[0], slope_AUC_time[1])]
    #     self.slope_data = self.slope_data.groupby(group_col, as_index=False).apply(lambda group: group.assign(slope_AUC = np.trapz(group['slope'], group['Time']))).reset_index(drop=True)
    #     self.slope_data['slope_AUC_mean'] = self.slope_data.groupby([i for i in group_col if i != 'Replicate'])['slope_AUC'].transform('mean')
    #     self.slope_data['slope_AUC_std'] = self.slope_data.groupby([i for i in group_col if i != 'Replicate'])['slope_AUC'].transform('std')
    #     # self.slope_data = self.slope_data.drop_duplicates(subset = [i for i in group_col if i != 'Replicate']).reset_index(drop = True)
