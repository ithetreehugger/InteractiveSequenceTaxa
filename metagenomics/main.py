import base64
import io
import itertools
from typing import List, Dict, Tuple

import pandas as pd

from bokeh.layouts import column, widgetbox, row
from bokeh.models import ColumnDataSource, TableColumn, TapTool, GlyphRenderer, Glyph, CustomJS, Button, FileInput
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import DataTable, Slider, Dropdown
from bokeh.palettes import Set3
from bokeh.events import Tap
from bokeh.transform import stack

# The expected columns from the sequence_taxa files 
original_columns = ['##Sequence_ID', 'Hit_Coordinates', 'NCBI_Taxon_ID', 'Taxon_Rank', 'Taxon_Name', 'Probability_Mass',
                 'Markers_Hit']

# The taxa ranks that will be display-able
taxa_rank_order = ["species", "genus", "family", "order", "class", "phylum", "no rank"]

# Reading in the lineage data as a dataframe
lineage_df = pd.read_csv("lineages-2019-12-08-abridged2.csv")

# global variable for storing filters in the form of a lambda and a value, which are later applied to the active input
source_filter = {}

# generate the source columns for the bokeh data table
columns: List[TableColumn] = [TableColumn(field=k, title=k) for k in original_columns]
# Storage for the data related to all input files; starts empty
inputs: Dict = {}

  
def getActiveSource():
    """
    Getter function for the bokeh source object of the active input
    """
    return inputs[active_input]['source']

  
def getActiveDf():
    """
    Getter function for the pandas dataframe object of the active input
    """
    return inputs[active_input]['df']


def getActiveTable():
    """
    Getter function for the bokeh data table object of the active input
    """
    return inputs[active_input]['table']

  
def generateLineageDf(df: pd.DataFrame):
    """
    Filter global lineage dataframe for only the taxon ids in the given dataframe, df 
    """
    ids = df['NCBI_Taxon_ID']
    return lineage_df[lineage_df.tax_id.isin(ids)]

  
def initDataSets():
    """
    initialize a dataset for the first time. If the dataset has already been initialized, this will do nothing.
    """
    global inputs
    for name, d in inputs.items():
        if "source" in d.keys():
            continue
        df: pd.DataFrame = inputs[name]['df']
        source = ColumnDataSource(df)
        inputs[name]['source']: ColumnDataSource = source
        inputs[name]['table']: DataTable = DataTable(source=source, columns=columns)
        inputs[name]['lineages'] = {'all':generateLineageDf(df)}
        inputs[name]['table'].sizing_mode = "stretch_both"


# global variable to store the currently selected input        
active_input = None


def activateInput(name: str) -> Tuple[DataTable, ColumnDataSource]:
    """
    activates the input given by name. name must exist in the global inputs dictionary
    """
    global active_input
    if active_input is None:
        return False
    if name not in list(inputs.keys()):
        raise IOError(f"No input provided for {name}")
    active_input = name
    return True


# Create the probability Mass slider.
prob_slider = Slider(start=0.00, end=1.00, step=.01, value=0.5, title="Probability Mass")
# Create the taxa rank drop down menu
taxa_menu = Dropdown(label="Taxon Rank", menu=taxa_rank_order, value=taxa_rank_order[-1])
# create the file choose 
file_chooser = FileInput(id="Load Sequence Taxa Files", accept=".txt,.csv,.tsv")
# create the input selection drop down menu
input_menu = Dropdown(label="Inputs", menu=list(inputs.keys()))
# create the reset button
reset_button: Button = Button(label="Reset Settings", button_type="warning")

# specify that we want all the following items in a single row    
top_row = row(prob_slider, taxa_menu, input_menu, reset_button, file_chooser)


# create a widgetbox (currently empty) to eventually store the data table
data_box = widgetbox(sizing_mode='stretch_both')

# If there is already an input during startup, activate it (only used for testing, really)
if (activateInput(active_input)):
    data_table = getActiveTable()
    children = data_box.children
    if len(children) == 0:
        children.insert(0, data_table)
    else:
        children[0] = data_table

def generateFreqDf(named_sources):
    """
    Generate a dataframe containing the frequency of each taxa member for the currently selected taxa.
    This not only counts the number of members of each taxa rank, but also aggregates the number of sub-rank members
    which fall within the current rank-member category.
    """
    frames = []
    for name, data in inputs.items():
        first_time = len(data['lineages']) == 1
        if needs_update() or first_time:
            source: ColumnDataSource = data['source']
            source_df = pd.DataFrame(source.data)
            tax_ids = source_df['NCBI_Taxon_ID']
            lineages = data['lineages']['all']
            lin_subset = lineages[lineages['tax_id'].isin(tax_ids)]
            for taxa in taxa_rank_order:
                counts = lin_subset[taxa].value_counts()
                freq_df = pd.DataFrame(counts).transpose()
                freq_df['files'] = name
                data['lineages'][taxa] = freq_df


        # Get the dataframe
        rank_freq: pd.DataFrame = data['lineages'][live_taxa_val]
        frames.append(rank_freq)
    freq_df = pd.concat(frames, ignore_index=True, sort=False)
    freq_df = freq_df.reset_index(drop=True)
    return freq_df


# global variable to hold the stacked graph data sources
stack_source = None

# Create the drop down menu which holds the members of the current rank
#   (or sub members, if we are filtering on a specifc member)
drill_down = Dropdown(label="Drill Down", menu=[])

def click_bar_stack(event):
    """
    Was hoping to allow drill-down based on clicking on the stacked graph; didn't get it working, 
    using dropdown menu instead
    """
    selected = stack_source.selected.indices


# store the last selected drill-down menu item
last_drill_down = ""
# store the last selecgted probability mass value
last_prob = ""
# store the last selected taxa rank value
last_rank = ""
# store the current drill-down sub-member value
live_drill_val = ""


def needs_update():
    """
    Update the data; ONLY if it needs to be updated. If nothing has changed, don't update.
    """
    update = False
    for last, cur in zip([last_prob, last_drill_down, last_rank], [prob_slider.value, live_drill_val, live_taxa_val]):
        if last == cur:
            continue
        else:
            return True
    return update

def updateStackedGraph():
    """
    Generate a new stacked graph and replace the old one. 
    First, get the frequency dataframes for each inputs lineages
    Then, normalize each lineage frequency to represent proportion out of 100% instead of count.
    Then, create each stacked bar, with alternating color pallete
    
    Lastly, this is usually the last graphical update to occur, so update the "last_" values
    """
    freq_df = generateFreqDf(inputs)

    # Normalize the frequency to be out of 100 percent
    non_files_cols = freq_df.columns.tolist()
    non_files_cols.remove('files')
    float_cols_df = freq_df[non_files_cols]
    norm_df = float_cols_df.apply(normalize_df, axis=1)
    norm_df = (norm_df * 100)
    # replace original values with normalized
    freq_df[non_files_cols] = norm_df
    # set NaN to 0
    freq_df = freq_df.fillna(0)
    # Remove zero-only columns
    freq_df = freq_df.loc[:, (freq_df != 0).any(axis=0)]

    tools = ['save', 'hover', 'pan', 'box_zoom', 'wheel_zoom', 'reset']
    drill_val = live_drill_val if live_drill_val in drill_down.menu else ""
    caps_taxa = live_taxa_val[0].upper() + live_taxa_val[1:]
    title = f"{caps_taxa} Proportion: {drill_val}"
    f = figure(y_range=freq_df['files'].tolist(), height=500, title=title, tools=tools,
               tooltips="$name: @$name%")
    f.xaxis.axis_label = "Cumulative Taxa Rank Proportion"
    f.yaxis.axis_label = "Input"

    stacks: List[str] = freq_df.columns.tolist()
    stacks.remove('files')
    drill_down.menu = stacks

    color_cyc = itertools.cycle(Set3[12])
    colors = [next(color_cyc) for i in range(len(stacks[:100]))]
    global stack_source
    stack_source = ColumnDataSource(freq_df)
    # TODO 100 is an arbitrary cutoff to speed up loading time
    f.hbar_stack(stacks[:100], y='files', height=0.9, fill_color=colors, line_color='black', source=stack_source)
    f.sizing_mode = "stretch_both"
    f.on_event(Tap, click_bar_stack)
    # Replace or add if not present.
    global stacked_graph_row
    stacked_graph_row.children = [f]
    # Now that we're done, update the last_ values
    global last_drill_down
    last_drill_down = live_drill_val
    global last_prob
    last_prob = prob_slider.value
    global last_rank
    last_rank = live_taxa_val


def normalize_df(df: pd.Series):
    """
    normalize the df so each row is a fract out of 1
    """
    df_sum = df.sum()
    res: pd.Series = df / df_sum
    res = res.apply(lambda x: x if x > .0009 else 0.0)
    return res


def updateAllSources(**kwargs):
    """
    Update all the sources. If a drill down menu was selected, filter on the dril down sub-member selected before updating
    """
    # Drill down was selected; lower the rank by one
    global last_drill_down
    global live_taxa_val
    # don't continue if the drilldown val is the same as last, or empty, or we are already as the lowest taxa rank
    if live_drill_val not in [last_drill_down, "", None] and live_taxa_val != taxa_rank_order[0]:
        # get the next lowest taxa rank
        tax_idx = taxa_rank_order.index(live_taxa_val)-1
        # update the taxa_menu
        live_taxa_val = taxa_rank_order[tax_idx]
        # set the last drill down to current
        last_drill_down = live_drill_val
        ### NOTE: This is a bit sketchy; intentionally "lie" about the last_prob_value to force it to update;
        #   otherwise, you end up with the frequency df not updating, and using the last known taxa freq df,
        #   which is incorrect. It gets set properly at the end of updating the stack grahp, so it should only
        #   impact the update round its set in. Worst-case should be that you end up duplicating some calculations
        #   All values should still be correct.
        global last_prob
        last_prob = ""

    for name, dct in inputs.items():
        update_source(dct)

# global boolean flag to know if the input has changed or not
input_changed = False

def update_source(data: Dict=None) -> None:
    """
    update a source. This is done by applying any sources in the source filter for the given data
    """
    cur_df: pd.DataFrame = data['df'] if data is not None else getActiveDf()
    src = data['source'] if data is not None else getActiveSource()
    mask = None
    for filter_name, mask_vals in source_filter.items():
        mask_lambda = mask_vals[0]
        mask_val = mask_vals[1]

        partial_mask = mask_lambda(cur_df, mask_val)
        if mask is None:
            mask = partial_mask
        else:
            merged_mask = mask & partial_mask
            mask = merged_mask
    cur_df = cur_df[mask] if mask is not None else cur_df


    # only if the drilldown value is in the menu, and the taxa rank is in the menu, and the taxa rank isn't the top level
    if (live_drill_val in drill_down.menu or input_changed) and live_taxa_val in taxa_menu.menu and live_taxa_val != taxa_rank_order[-1]:
        # We have already decremented the taxa menu, now get the super tax to filter by the selected member
        tax_idx = taxa_rank_order.index(live_taxa_val) + 1
        super_taxa = taxa_rank_order[tax_idx]

        # filter for only the lineages we are tracking
        tax_ids = cur_df['NCBI_Taxon_ID']
        lineages = data['lineages']['all']
        lin_subset = lineages[lineages['tax_id'].isin(tax_ids)]
        # filter for the drill down menu
        filtered_tax_ids = lin_subset[lin_subset[super_taxa] == live_drill_val].tax_id.tolist()
        cur_df = cur_df[pd.to_numeric(cur_df['NCBI_Taxon_ID']).isin(filtered_tax_ids)]


    if (len(cur_df) == 0):
        cur_df.append(['NO_DATA']*len(cur_df.columns))
    subset = ColumnDataSource(cur_df)
    src.data = subset.data


def prob_slider_callback(attr, old, new):
    """
    function triggered when the probabiltiy mass slider is changed; 
    updates the source filter, and triggers an update if needed
    """
    mask = lambda df, x: df.Probability_Mass >= x
    source_filter['prob'] = (mask, new)
    updateIfNeeded()

# global flag to block and update if we are setting multiple values programmatically.
manual_change = False

def updateIfNeeded():
    """
    Checks if an update is necessary before triggering an update;
    """
    if needs_update() and not manual_change:
        updateAllSources()
        updateStackedGraph()


# Currently active taxa value
live_taxa_val = taxa_rank_order[-1]
def taxa_menu_callback(attr, old, new):
    """
    function to trigger when the taxa rank drop down menu changes.
    update if needed.
    """
    global live_taxa_val
    live_taxa_val = new
    label = "Taxon Rank"
    taxa_menu.label = f"{label}: {new}"
    global live_drill_val
    global last_drill_down
    last_drill_down = ""
    live_drill_val = None
    updateIfNeeded()

# Initialize the probability mass value to .5
live_prob = .5
def input_menus_callback(attr, old, new):
    """
    function to trigger when the input selection is changed. Updates the data table
    """
    if old == new:
        return

    label = "Input"
    input_menu.label = f"{label}: {new}"
    global input_changed
    input_changed = True

    activateInput(new)
    updateAllSources()
    children = data_box.children
    if len(children) == 0:
        children.insert(0, getActiveTable())
    else:
        children[0] = getActiveTable()

    input_changed = False

    
def drill_down_callback(attr, old, new):
    """
    Function triggered when the drill-down menu is updated. updates if needed.
    """
    global live_drill_val
    live_drill_val = new

    drill_down.label = "Filter on Sub-Rank"

    updateIfNeeded()



def reset_settings():
    """
    Reset the values of all UI objects to default values;
    probability slides get .5
    taxa rank gets the highest taxa rank
    active input set to first input
    drill-down menu reset to none
    """
    global manual_change
    global last_prob
    global last_drill_down
    manual_change = True
    last_prob = ""
    last_drill_down = ""
    drill_down.value = ""
    taxa_menu_callback('value', '', taxa_rank_order[-1])
    prob_slider.value = .5
    first_input = list(inputs.keys())[0]
    input_menus_callback('value', '', first_input)
    global live_taxa_val
    live_taxa_val = taxa_rank_order[-1]
    input_menu.value = first_input
    prob_slider_callback('value', '', .5)
    manual_change = False
    updateIfNeeded()

def add_input_file(attr, old, new):
    """
    function triggered when a new input file is uploaded.
    decode the contents, read as a dataframe, add to the inputs, and initialize the data
    """
    contents = base64.standard_b64decode(new)
    df = pd.read_csv(io.BytesIO(contents), sep='\t')
    df = df[df.Taxon_Rank.isin(taxa_rank_order)]
    name = file_chooser.filename
    global input_menu
    input_menu.menu.append(name)
    global inputs
    inputs[name] = {"df":df}
    initDataSets()
    global active_input
    active_input = name
    if len(inputs.keys()) == 1:
        activateInput(name)
        start()
    else:
        reset_settings()

# Set the callback triggers for all the interactive UI elements
prob_slider.on_change('value', prob_slider_callback)
taxa_menu.on_change('value', taxa_menu_callback)
file_chooser.on_change('value', add_input_file)
input_menu.on_change('value', input_menus_callback)
drill_down.on_change('value', drill_down_callback)
reset_button.on_click(reset_settings)

# ensure the stacked graph fits the screen
stacked_graph_row = column(sizing_mode="stretch_both")

# Set up the entire web-page as a a single column, containing a top row, and a bottom row
page = column(top_row, data_box, row(stacked_graph_row, drill_down))

def start():
    """
    Start function first called by bokeh
    """
    global manual_change
    # Turn off auto-update until we are down setting all, then call manually
    manual_change = True
    # Manually trigger callback for the first time
    taxa_menu_callback('value', 'None', 'class')
    # Manually trigger callback for the first time
    input_choice = list(inputs.keys())[0]
    input_menus_callback('value', 'None', input_choice)
    # Manually trigger callback for the first time
    prob_slider_callback('value', 'None', .5)
    manual_change = False
    updateIfNeeded()

# Set the web page size mode
page.sizing_mode = "stretch_both"

# add the page to the current bokeh document
curdoc().add_root(page)
