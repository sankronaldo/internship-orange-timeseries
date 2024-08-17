# RDataReader Widget

The RDataReader widget is designed to read and import data from `.RData` files into the Orange framework. It supports loading data from local files and allows users to configure column roles such as features, targets, and meta attributes.

![](../images/sankarsh-widgets/readRdata/rdata.png)

## Parameters

### Source Type
- **Source Type**: Dropdown menu to select the data source. Options include:
  - **File**: Load data from a local `.RData` file.
  - **URL**: Load data from a URL (not yet implemented for `.RData` files).

### Filename
- **Filename**: Path to the `.RData` file. Users can select the file using the file picker dialog.

### URL
- **URL**: URL to load the `.RData` file from (not implemented).

### Column Role Configuration
- **Column Role Configuration**: Users can configure the role of each column in the dataset (Feature, Target, Meta, or Skip) using a dropdown menu for each column. The widget displays the column name, type, role, and unique values.

## Inputs
None

## Outputs
- **Data**: An Orange `Table` containing the imported dataset.

