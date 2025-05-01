import os
import numpy as np

class PathConfig:
    def __init__(self, g_dir=None,g2_dir=None):
        self.save_dir = os.path.join('analysis','lhb_data')
        if g_dir is None:
            self.g_dir = os.path.join('data','osfstorage') # modify 'data' for any path if necessary
        if g2_dir is None:
            self.g2_dir = os.path.join('data','googledrive') # modify 'data' for any path if necessary


import os
import numpy as np

def save_full_figure_data( sectioned_data_dict, stats_dict=None, base_filename='figure', folder='figure_data', stats_names=None):
    """
    Save a figure and multiple related data blocks (including optional stats and stat names) into a single .txt file.

    Parameters:
    - fig: matplotlib figure object
    - sectioned_data_dict: dict where keys are section names, values are (data_array, header_list)
    - stats_dict: optional, dict where keys are stat section names, values are (stats_array, header_list)
    - base_filename: base filename (e.g., 'figure_1d')
    - folder: save location
    - stats_names: optional, dict where keys match stats_dict keys, and values are lists of stat names (strings)
    """

    os.makedirs(folder, exist_ok=True)

    # # Save figure
    # fig_filename = os.path.join(folder, f"{base_filename}.pdf")
    # fig.savefig(fig_filename, bbox_inches='tight')
    # print(f"Saved figure to {fig_filename}")

    # Save data and stats
    data_filename = os.path.join(folder, f"{base_filename}.txt")
    with open(data_filename, 'w') as f:
        # Save regular figure data
        for section_name, (data_array, header_list) in sectioned_data_dict.items():
            f.write(f"# {section_name}\n")
            f.write('\t'.join(header_list) + '\n')
            if data_array.ndim == 1:
                data_array = data_array[:, np.newaxis]
            np.savetxt(f, data_array, fmt='%.6f', delimiter='\t')
            f.write("\n\n")

        # Save stats if provided
        if stats_dict is not None:
            for stat_section_name, (stat_array, stat_headers) in stats_dict.items():
                f.write(f"# {stat_section_name}\n")
                
                # Write the name of the test(s) as comments
                if stats_names is not None and stat_section_name in stats_names:
                    test_names = stats_names[stat_section_name]
                    for idx, test_name in enumerate(test_names):
                        f.write(f"# Group {idx}: Test used = {test_name}\n")

                # Write header and stat data
                f.write('\t'.join(stat_headers) + '\n')
                if stat_array.ndim == 1:
                    stat_array = stat_array[:, np.newaxis]
                np.savetxt(f, stat_array, fmt='%.6f', delimiter='\t')
                f.write("\n\n")

    print(f"Saved data (and stats) to {data_filename}")


def save_figure_data( data_list, headers_list, base_filename, panel_names=None, folder='figure_data'):
    """
    Save figure data arrays and the figure itself.

    Parameters:
    - fig: matplotlib figure object
    - data_list: list of arrays to save (one array per panel)
    - headers_list: list of headers (one list of strings per array)
    - base_filename: base name for saving (e.g., 'figure_1d')
    - panel_names: list of names for each panel (e.g., ['left', 'right']), optional
    - folder: folder to save everything (default = 'figure_data')
    """
    os.makedirs(folder, exist_ok=True)

    if panel_names is None:
        panel_names = [f"panel{idx+1}" for idx in range(len(data_list))]

    assert len(data_list) == len(headers_list) == len(panel_names), "Mismatch in data, headers, or panel names"

    # Save each data array
    for data, headers, panel_name in zip(data_list, headers_list, panel_names):
        filename = os.path.join(folder, f"{base_filename}_panel_{panel_name}.txt")
        if data.ndim == 1:
            data = data[:, np.newaxis]  # Ensure 2D for saving
        np.savetxt(filename, data, header='\t'.join(headers), fmt='%.6f')
        print(f"Saved data to {filename}")


def save_figure_pdf(fig,base_filename,folder='figure_data'):
    # Save the figure itself
    fig_filename = os.path.join(folder, f"{base_filename}.pdf")
    fig.savefig(fig_filename, bbox_inches='tight')
    print(f"Saved figure to {fig_filename}")

class FigureWithData:
    def __init__(self, fig):
        self.fig = fig
        self.data_sections = {}
        self.stats_sections = {}
        self.stats_names = {}

    def add_data(self, section_name, data_array, headers):
        self.data_sections[section_name] = (data_array, headers)

    def add_stats(self, section_name, stats_array, headers, test_names=None):
        self.stats_sections[section_name] = (stats_array, headers)
        if test_names is not None:
            self.stats_names[section_name] = test_names

    def save(self, base_filename, folder_data='figure_data', folder_figures='figures'):
        """
        Save figure as .pdf (in folder_figures)
        Save data and stats as .txt (in folder_data)
        """

        # Create folders if needed
        os.makedirs(folder_data, exist_ok=True)
        os.makedirs(folder_figures, exist_ok=True)

        # --- Save figure ---
        fig_filename = os.path.join(folder_figures, f"{base_filename}.pdf")
        self.fig.savefig(fig_filename, bbox_inches='tight')
        print(f"Saved figure to {fig_filename}")

        # --- Save data and stats ---
        data_filename = os.path.join(folder_data, f"{base_filename}.txt")
        with open(data_filename, 'w') as f:
            # Save data sections
            for section_name, (data_array, headers) in self.data_sections.items():
                f.write(f"# {section_name}\n")
                f.write('\t'.join(headers) + '\n')
                if data_array.ndim == 1:
                    data_array = data_array[:, np.newaxis]
                np.savetxt(f, data_array, fmt='%.6f', delimiter='\t')
                f.write("\n\n")

            # Save stats sections
            for section_name, (stat_array, stat_headers) in self.stats_sections.items():
                f.write(f"# {section_name}\n")
                if section_name in self.stats_names:
                    for idx, test_name in enumerate(self.stats_names[section_name]):
                        f.write(f"# Group {idx}: Test used = {test_name}\n")
                f.write('\t'.join(stat_headers) + '\n')
                if stat_array.ndim == 1:
                    stat_array = stat_array[:, np.newaxis]
                np.savetxt(f, stat_array, fmt='%.6f', delimiter='\t')
                f.write("\n\n")

        print(f"Saved data and stats to {data_filename}")
