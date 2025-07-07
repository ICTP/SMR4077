# Based on https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/weather.ipynb

import cudf
import dask, dask_cudf
import os
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from dask.diagnostics import ProgressBar
import cuspatial
import geopandas
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

# Simpler Multi-GPU ETL using Dask

# A major focus of the RAPIDS project is easier scaling: up and out.

# The dask-cuda project automatically handles configuring Dask worker processes to make use of available GPUs
# The dask-cudf supports a variety of common ETL operations and friendlier parallel IO.

# Here we demonstrate just how simple parallel processing is with RAPIDS, and 
# how you can scale your data science work to multiple GPUs with ease.

def main():
    os.environ["DASK_DISTRIBUTED__COMM__UCX__NVLINK"] = "True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__NET_DEVICES"] = "ib0"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__TCP"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__NVLINK"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__RDMACM"]="True"
    os.environ["UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES"]="cuda"
    os.environ["UCX_MEMTYPE_CACHE"]="n"
    # Use dask-cuda to start one worker per GPU on a single-node system
    # When you shutdown this notebook kernel, the Dask cluster also shuts down.
    current_dir=os.environ.get('SLURM_SUBMIT_DIR')
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=[0, 1, 2, 3], 
                               n_workers=4, 
                               threads_per_worker=8,
                               protocol="ucx", 
                               interface="ib0", 
                               enable_tcp_over_ucx=True, 
                               enable_infiniband=True, 
                               enable_nvlink=True, 
                               enable_rdmacm=True,
                               rmm_pool_size="40GB",)

    client = Client(cluster)
    client.wait_for_workers(4)
    # print cluster and client info
    print("\ncluster:\n", cluster, flush=True)
    print("\nclient:\n",  client,  flush=True)
    # Ok, we've got a cluster of GPU workers. 
    # Notice also the link to the Dask status dashboard.
    # It provides lots of useful information while running data processing tasks.
    # On Leonardo compute node we cannot download the dataset
    # We need to have pre-downloaded data!
    # Here we assume that data have been already downloaded and the files are in the following directory
    data_dir="/leonardo_work/tra25_ictp_rom/DAY8data/weather/data/"
    # Notice that the CSV files don't have headers, we specify column names manually
    names = ["station_id", "date", "type", "val"]
    # Moreover, there are a lot of features and fields, but only the first 4 are relevant for us
    usecols = names[0:4]
    # Wait... there are many weather files.
    # To read all these files in, we can simply use dask_cudf.read_csv
    # It automatically splits files into chunks that can be processed serially when needed, so you're less likely to run out of memory.
    # When you call dask_cudf.read_csv, Dask reads metadata for each CSV file and tasks workers with lists of filenames & byte-ranges. 
    # The workers are responsible for loading with cuDF's GPU CSV reader.
    # Note: compressed files are not splittable on read, but you can repartition them downstream.
    weather_ddf = dask_cudf.read_csv(data_dir+'*.csv.gz', names=names, usecols=usecols, compression='gzip', blocksize=None)
    # For this dataset, multiple types of weather observations are in the same files, and each carries a different units of measure:
    # PRCP 	Precipitation (tenths of mm)
    # SNWD 	Snow depth (mm)
    # TMAX 	tenths of degrees C
    # TMIN 	tenths of degrees C
    # There are more even more observation types, each with their own units of measure, but I won't list them all. 
    # Here we are going to focus specifically on precipitation.
    # The type column tells us what kind of weather observation each record represents.
    # Ordinarily, we might use query to filter out subsets of records and apply different logic to each subset.
    # However, query doesn't support string datatypes yet.
    # Instead, you can use boolean indexing.
    # For numeric types, Dask with cuDF works mostly like regular Dask.
    # For instance, we can define new columns as combinations of other columns.
    # As an example, suppose we wanto to convert the Precipitation (PRCP) from mm to inches
    precip_index = weather_ddf['type'] == 'PRCP'
    precip_ddf = weather_ddf[precip_index]
    # convert 10ths of mm to inches
    mm_to_inches = 0.0393701
    precip_ddf['val'] = precip_ddf['val'] * 1/10 * mm_to_inches
    # In our case, the first partition represents weather data from 2000.
    print("\nprecip_ddf.get_partition(1).head():\n", precip_ddf.get_partition(1).head(), flush=True)
    # Note: Calling .head() will read the first few rows, usually from the first partition.
    # Beware in your own analyes, that you .head() from partitions that you haven't already filtered everything out of!
    # Ok, we have a lot of weather observations. Now what?
    # For some reason, residents of particular cities like to lay claim to having the best, or the worst of something.
    # For Los Angeles, it's having the worst traffic.
    # New Yorkers and Chicagoans argue over who has the best pizza.
    # West Coasters argue about who has the most rain.
    # Well... as a longtime Atlanta resident suffering from humidity exhaustion.
    # With all the spring showers, is Atlanta the new Seattle?
    # How can we test this theory?
    # We've already created precip_df, which is only the precipitation observations, but it's for all 100k weather stations!
    # Most of them nowhere near Atlanta, and this is time-series data, so we'll need to aggregate over time ranges.
    # To get down to just Atlanta and Seattle precipitation records, we have to...
    # 1. Extract year, month, and day from the compound "date" column, so that we can compare total rainfall across time.
    # 2. Load up the station metadata file.
    # 3. There's no "city" in the station metadata, so we'll do some geo-math and keep only stations near Atlanta and Seattle.
    # 4. Use a Groupby to compare changing precipitation patterns across time
    # 5. Use inner joins to filter the precipitation dataframe down to just Atlanta & Seattle data.
    # Let's start with the first step:
    # 1. Extracting Finer Grained Date Fields
    # We can use cuDF's to_datetime function to map our date column into separate date parts.
    # Dask's map_partitions function applies a given Python function to all partitions of a distributed DataFrame or Series.
    # When you do this on a dask_cudf DataFrame or Series, your input is a cuDF object.
    # convert date column to a series of datetime objects
    dates = precip_ddf['date'].map_partitions(cudf.to_datetime, format='%Y%m%d', meta=("date", "datetime64[ns]"))
    #assign new columns to their respective date parts
    precip_ddf['year'] = dates.dt.year
    precip_ddf['month'] = dates.dt.month
    precip_ddf['day'] = dates.dt.day
    print("\nprecip_ddf.head():\n", precip_ddf.head(), flush=True)
    # The map_partitions pattern is also useful whenever there are cuDF specific functions without a direct mapping into Dask.
    # 2. The station metadata file is the following
    fn = data_dir+'ghcnd-stations.txt'
    # Note that that's no CSV file! It's fixed-width!
    # That's annoying because we don't have a reader for it.
    # We could use CPU code to pre-process the file, making it friendlier for loading into a DataFrame.
    # But RAPIDS is about end-to-end data processing without leaving the GPU.
    # This file is small enough that we can handle it directly with cuDF on a single GPU.
    # Here's how to cleanup this metadata using cuDF and string operations.
    # There are no '|' chars in the file. Use that to read the file as a single column per line
    station_df = cudf.read_csv(fn, sep='|', quoting=3, names=['lines'], header=None)
    # Above, quoting=3 handles misplaced quotes in the `name` field 
    # We can use normal DataFrame .str accessor, and chain operators together
    station_df['station_id'] = station_df['lines'].str.slice(0, 11).str.strip()
    station_df['latitude']   = station_df['lines'].str.slice(12, 20).str.strip()
    station_df['longitude']  = station_df['lines'].str.slice(21, 30).str.strip()
    station_df = station_df.drop('lines', axis=1)
    print("\nstation_df.head():\n", station_df.head(), flush=True)
    # Managing Memory
    # While GPU memory is very fast, there's less of it than host RAM.
    # It's a good idea to avoid storing lots of columns that aren't useful for what you're trying to do.
    # Especially when they're strings.
    # For example, for the station metadata, there are more columns than we parsed out above.
    # In this workflow we only need station_id, latitude, and longitude, so we skipped parsing the rest of the columns.
    # We also need to convert latitude and longitude from strings to floats, 
    # Finally, we need to convert the single-GPU DataFrame to a Dask DataFrame that can be distributed across workers.
    # We can cast string columns to numerics as follows
    station_df['latitude'] = station_df['latitude'].astype('float')
    station_df['longitude'] = station_df['longitude'].astype('float')
    # Let's see if the result is as expected by saving to file in the test.csv
    station_df.head(20).to_csv("test.csv", index = False)
    # 3. Filtering Weather Stations by Distance
    # We will be using cuSpatial to get the Haversine Distance and figure out which stations are within a given distance from a city.
    # For this scenario, we've manually looked up Atlanta and Seattle's city centers.
    # Then, we will fill cudf.Series with their latitude and longitude values.
    # Finally, we can call a cuSpatial function to compute the distance between each station and each city.
    # Let's create a cuSpatial GeoSeries with the station data
    stations = cuspatial.GeoSeries.from_points_xy(station_df[['longitude','latitude']].interleave_columns())
    # Fill new GeoSeries with Atlanta lat/lng
    station_df['atlanta_lat'] = 33.7490
    station_df['atlanta_lng'] = -84.3880
    atl = cuspatial.GeoSeries.from_points_xy(station_df[['atlanta_lng','atlanta_lat']].interleave_columns())
    # Compute distance from each station to Atlanta
    station_df['atlanta_dist'] = cuspatial.haversine_distance(stations, atl)
    # Fill new GeoSeries with Seattle lat/lng
    station_df['seattle_lat'] = 47.6219
    station_df['seattle_lng'] = -122.3517
    stl = cuspatial.GeoSeries.from_points_xy(station_df[['seattle_lng','seattle_lat']].interleave_columns())
    # Compute distance from each station to Seattle
    station_df['seattle_dist'] = cuspatial.haversine_distance(stations, stl)
    atlanta_stations_df = station_df.query('atlanta_dist <= 25')
    seattle_stations_df = station_df.query('seattle_dist <= 25')
    # Inspect the results:
    print(f"\nAtlanta Stations:\n{len(atlanta_stations_df)}\n", flush=True)
    print(f"\nSeattle Stations:\n{len(seattle_stations_df)}\n", flush=True)
    print("\natlanta_stations_df.head():\n", atlanta_stations_df.head(), flush=True)
    print("\nseattle_stations_df.head():\n", seattle_stations_df.head(), flush=True)
    # 4. Grouping & Aggregating by Time Range
    # Now, we can use a groupby to sum the precipitation for station and year.
    # That'll allow the join to proceed faster and use less memory.
    # One total precipitation record per station per year is relatively small, and we're going to need to graph this data.
    # We'll go ahead and compute() the result, asking Dask to aggregate across the data, bringing the results back to the client.
    # The result is a single GPU cuDF DataFrame.
    # Note that with Dask, data is partitioned and distributed across multiple workers.
    # Some operations require that workers "shuffle" data from their partitions back and forth across the network.
    # Today join, groupby, and sort operations can be fairly network constrained.
    # Distributed operators that require shuffling like joins, groupbys, and sorts work, albeit not as fast as we'd like.
    precip_year_ddf = precip_ddf.groupby(by=['station_id', 'year']).val.sum()
    precip_year_df = precip_year_ddf.compute()
    # Convert from the groupby multi-indexed DataFrame back to a normal DF which we can use with merge
    precip_year_df = precip_year_df.reset_index()
    # Note that we're calling compute again here.
    # This tells Dask to actually start computing the full set of processing logic defined thus far:
    # Read and decompress the gzipped files
    # Send to the GPU and parse
    # Filter down to precipitation records
    # Apply a conversion to inches
    # Sum total inches of rain per year per each of the weather stations
    # Combine and pull results a single GPU DataFrame on the client host
    # To wit... this will take some time.
    # 5. Using Inner Joins to Filter Weather Observations
    # We have separate DataFrames containing Atlanta and Seattle stations.
    # And we have our total precipitation grouped by station_id and year.
    # Computing inner joins can let us compute total precipitation by year for just Atlanta and Seattle.
    atlanta_precip_df = precip_year_df.merge(atlanta_stations_df, on=['station_id'], how='inner')
    print("\natlanta_precip_df.head():\n", atlanta_precip_df.head(), flush=True)
    seattle_precip_df = precip_year_df.merge(seattle_stations_df, on=['station_id'], how='inner')
    print("\nseattle_precip_df.head():\n", seattle_precip_df.head(), flush=True)
    # Lastly, we need to normalize the total amount of rain in each city by the number of stations which collected rainfall:
    # Seattle had twice as many stations collecting, but that doesn't mean more total rain fell!
    atlanta_rain = atlanta_precip_df.groupby(['year']).val.sum()/len(atlanta_stations_df)
    print("\natlanta_rain.head():\n", atlanta_rain.head(), flush=True)
    seattle_rain = seattle_precip_df.groupby(['year']).val.sum()/len(seattle_stations_df)
    print("\nseattle_rain.head():\n", seattle_rain.head(), flush=True)
    # Visualizing the Answer
    plt.close('all')
    plt.rcParams['figure.figsize'] = [20, 10]
    fig, ax = subplots()
    atlanta_rain.sort_index().to_pandas().plot(ax=ax)
    seattle_rain.sort_index().to_pandas().plot(ax=ax)
    ax.legend(['Atlanta', 'Seattle'])
    plt.savefig(current_dir + "/res.png", bbox_inches='tight')
    # Results
    # It looks like at least for roughly the last 20 years, it rains more by volume in Atlanta than it does in Seattle.
    # But as usual the answer raises additional questions:
    # 1. Without singling out Atlanta and Seattle, which city actually has the most precipitation by volume?
    # 2. Why is there such a large increase in observed precipitation in the last 10 years?
    # 3. While it rains more frequently in Seattle (just not as hard), it also mists a lot in Seattle. 
    # How often is it just "misty", but not really raining?
    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

