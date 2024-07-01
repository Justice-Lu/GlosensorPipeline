import pandas as pd 
import numpy as np 
import sys 
import os 
import copy
from csv import reader
import csv 
import plotly.express as px
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import random 

def getFileNames(file_dir):
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

def ReadTableData(file_dir):
    '''
    ReadTableData takes in .csv files exported from tableview of the MARS program and combines plates with blank reading as Time_0
    input:
    file_dir: paths to directory where the files lives. 
    (Note that file names should be in the format of data_initial_platenumber_blankoptional ex. 12241994_JL_1_Blank)
    '''
    files = getFileNames(file_dir)
    combine_data = {}
    for file in files:
        data = pd.DataFrame()
        
        if 'blank' in file.lower():
            plate = f"Plate_{file.split('.')[0].split('_')[2]}_blank" #Updates the plate # based on the 3rd index of '_'
        else:
            plate = f"Plate_{file.split('.')[0].split('_')[2]}" #Updates the plate # based on the 3rd index of '_'

        with open(os.path.join(file_dir + file), 'r') as file:
            reader = csv.reader(file)
            
            # Skip the first 7 lines
            for _ in range(7):
                next(reader)
            for line in reader:
                try: 
                    if line[0] == '':
                        col_name = line[1:]
                    if line[0][0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                        row_data = pd.DataFrame([line])
                    data = pd.concat([data,row_data])
                except: 
                    continue
        combine_data[plate] = data

    
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
        plate_data.insert(1, 'Plate', k)
        
        all_plate_data = pd.concat([all_plate_data, plate_data])
    
    # Convert Position into Rows and Columns identifiers  
    all_plate_data.insert(1, 'Row', all_plate_data['Position'].str.split('(\d+)', expand=True)[0]) 
    all_plate_data.insert(1, 'Column', all_plate_data['Position'].str.split('(\d+)', expand=True)[1]) 
    all_plate_data['Column'] = all_plate_data['Column'].astype(int)
    
    return all_plate_data 
            

def ReadPlates(file_dir):
    
    """
    file_dir is the directory which the csv data is stored at 

    Returns the dataframe holding the 96-well data
    """    
    
    files = getFileNames(file_dir)
# Using open to read csv files line by line. 
    data = pd.DataFrame(columns=['Position','Row','Column', 'Time', 'Value','Plate'])
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
                        row_data = pd.DataFrame(columns=data.columns)
                        for i in range(1,len(line)):
                            if not line[i] == '': #Skip if the cell has no record
                                row_data = pd.concat([row_data, pd.DataFrame([[line[0]+str(i),
                                                                          line[0], 
                                                                          i,
                                                                          interval,
                                                                          float(line[i]), 
                                                                          int(plate)]],
                                                                        columns=data.columns)])
                        data = pd.concat([data,row_data])
#     Adds numerical index to dataframe
    data.index = range(len(data))
    return data       

def getGroupColor(group):
    """
    Takes in vector of groups, and assign colors for every group. 
    Returns vector of groups with corresponding color dataframe
    """
    group_color = {}
    color_index = 0
    for i in group:
        group_color[i] = px.colors.qualitative.Plotly[color_index]
        color_index += 1
    return  group_color
                   
    
def getDataSummary(data, param=['Group', 'Odor', 'Odor concentration (uM)']):
    """
    Calculate the Mean and Standard Deviation of the data 
    Returns a summary table averaging same X and Y at a given time 
    More sub categories can be added, just create a variable and add additional for loops 
    """
    
    X = param[0]
    Y = param[1]
    Z = param[2]
    
    data_sum = pd.DataFrame(columns = ['Time','Plate',X,Y,Z,'Mean', 'Std'])

    for x in data[X].unique():
        for y in data[Y].unique():
            for z in data[Z].unique():
                temp = data[(data[X] == x) &
                            (data[Y] == y) &
                            (data[Z] == z)] 
                if (not temp.empty) & (data_sum[(data_sum[X] == x) & 
                                                (data_sum[Y] == y) & 
                                                (data_sum[Z] == z)].empty): # only proceed to append to data_sum if temp isn't empty                   
                    for time in data['Time'].unique():
                        data_sum = pd.concat([data_sum, pd.DataFrame([[time, 
                                                                       int(temp['Plate'].unique()[0]),
                                                                       x, #Group 
                                                                       y, #Odor
                                                                       z, #Odor concentration
                                                                       temp[temp['Time'] == time].Value.mean(), 
                                                                       temp[temp['Time'] == time].Value.std()]], 
                                                columns=data_sum.columns)])
    return(data_sum)
        

def addDataSum_AUC(data_sum, param=['Group', 'Odor', 'Odor concentration (uM)']):
    
    """
    Measures the Area Under Curve (AUC) of the given sample. 
    Simply the Sum of Mean value for every cell across 1-150 time intervals
    """
    
    X = param[0]
    Y = param[1]
    Z = param[2]
    
    for x in data_sum[X].unique():
        for y in data_sum[Y].unique():
            for z in data_sum[Z].unique():
                temp = data_sum[(data_sum[X] == x) & 
                                (data_sum[Y] == y) & 
                                (data_sum[Z] == z)]
                if not temp.empty: # only proceed to append to data_sum if temp isn't empty
                    for time in data_sum['Time'].unique():
                        data_sum.loc[(data_sum[X] == x) & 
                                     (data_sum[Y] == y) & 
                                     (data_sum[Z] == z) &
                                     (data_sum['Time'] == time), ['AUC']] = temp.Mean.sum()
    return(data_sum)                    

def normalize_by_t(data = None,
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
        time_col = [col for col in t_norm_data.columns if col.startswith('Time_')]
        
        if how == 'divide': 
            t_norm_data[time_col] = t_norm_data[time_col].div(t_norm_data[f'Time_{by_time}'], axis=0)
        elif how == 'subtract':
            t_norm_data[time_col] = t_norm_data[time_col].subtract(t_norm_data[f'Time_{by_time}'], axis=0)
        else: 
            print("Please specify 'how' as one of the options ['divide', 'subtract']")
        
        print(f'data normalized by Time_{by_time} with {how}')
        return t_norm_data
    
def addDataSum_Normalized(data_sum, param=['Odor concentration (uM)', 'Group']):
    
    """
    Normalize 
    Using Data_summary table, normalize the Mean by dividing stimulant by designated control (Rho-pCI)
    """

    X = param[0]
    Y = param[1]

    # for plate in data_sum['Plate'].unique():
    for x in data_sum[X].unique():
    #     for y in [group for group in list(data_sum[Y].unique()) if 'control' not in group]:
        for y in data_sum[Y].unique():
            temp = data_sum[(data_sum[X] == x) & 
                        (data_sum[Y].str.contains(y))]
            if not temp.empty: # only proceed to append to data_sum if temp isn't empty
                for time in data_sum['Time'].unique():
                    control_mean = temp[(temp[Y].str.contains('control')) & 
                                         (temp['Time'] == time)].Mean
                    control_std = temp[(temp[Y].str.contains('control')) & 
                                         (temp['Time'] == time)].Std
                    data_sum.loc[(data_sum[X] == x) & 
                            (data_sum[Y] == y) & 
                            (data_sum['Time'] == time), ['Normalized_Mean','Normalized_Std']] =  [[temp[(temp['Time'] == time) & 
                                                                                                         (temp[Y] == y)].Mean/control_mean],
                                                                                                  [temp[(temp['Time'] == time) &
                                                                                                        (temp[Y] == y)].Std/control_std]]

    return data_sum

def get_melted_AUC_slope_data(group_col: list, 
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

def plot_facet_line(df : pd.DataFrame,
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


def plot_Dose_curves(data_lists,
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

"""
THIS FUNCTION IS DEPRECATED AND NO LONGER USED
"""
# def Normalize_first_30_interval(data):
#     """
#     Since interval 0-30 are control before stimulation. 
#     Therefore the data should be normalized to the baseline of null stimulation. 
#     Create a dataframe baseline_average that holds the 96-well plate's 0-30 average 
#     """
#     baseline_data = copy.deepcopy(data.loc[data.time<31])
#     baseline_average = pd.DataFrame(columns=['position','row','column', 'value'])
#     for r in baseline_data.row.unique():
#         for c in baseline_data.column.unique():
#             average = baseline_data[(baseline_data.row==r) & (baseline_data.column==c)].value.mean()
#             baseline_average = pd.concat([baseline_average, pd.DataFrame([[r+str(c), 
#                                                         r,
#                                                         c, 
#                                                         average]], 
#                                                       columns=['position','row','column', 'value'])])

#     """
#     Divide the data table by the average base on every single cell 
#     time 1-30's normalized_value should be close to 0 
#     """
#     for r in data.row.unique():
#         for c in data.column.unique():
#             value = data[(data.row==r)&(data.column==c)].value
#             average = baseline_average[(baseline_average.row==r)&(baseline_average.column==c)].value
#             data.loc[(data.row==r)&(data.column==c),'Normalized_value'] = value/average
    
#     return data 

                    
                    
                    
                    
                    
                    
                    
                    
