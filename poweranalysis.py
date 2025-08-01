# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from settings import *

from pathlib import Path


# %%
class HighPowerSpecViolation:
    def __init__(self, vop_path: str = None, iop_path: str = None, power_path: str = None) -> None:
        # process input file
        if vop_path:
            print("Loading raw vop data.")
            self.vop = self.load_raw_csv(vop_path)
        if iop_path:
            print("Loading raw iop data.")
            self.iop = self.load_raw_csv(iop_path)
        if power_path:
            print("Loading raw power data.")
            self.power = self.load_raw_csv(power_path)

        if vop_path or iop_path or power_path:
            print("Getting testsuites.")
            self.testsuites = self.get_testsuites()
        else:
            raise ValueError("No files given.")

        if vop_path:
            print("Processing vop data.")
            self.vop_processed = self.process_vop_iop("voltage")
        if iop_path:
            print("Processing iop data.")
            self.iop_processed = self.process_vop_iop("current")
        if power_path:
            print("Processing power data")
            self.power = self.process_vop_iop("power")
            self.power_per_testsuite = self.get_power_per_testsuite()

            print("Computing energy per testsuite.")
            self.energy_per_testsuite = self.get_energy_per_testsuite()

        print("Getting pinname.")
        self.pinnames = self.get_pinnames()

        # removing raw iop and vop to save memory
        print("Deleting raw iop and vop.")
        del self.iop
        del self.vop

    @property
    def column_types(self) -> dict:
        return {
            "Seq": int,
            "PinName": str,
            "Site": int,
            "Type": str,
            "SampleRate": float,
            "Testsuite": str,
            "Measurement": str,
            "Sequencer": str,
            "StartTime": str,
            "Duration(us)": float,
            "SampleCount": int,
            "WaferID": str,
            "XCoord": int,
            "YCoord": int,
            "Min Data": float,
            "Max Data": float,
            "Avg Data": float,
            "WAVEFORM DATA": self.float_list,
        }

    def get_pinnames(self) -> list[str]:
        if hasattr(self, "vop") and hasattr(self, "iop"):
            vop_pinnames = self.vop["PinName"].unique()
            iop_pinnames = self.iop["PinName"].unique()

            if set(vop_pinnames) != set(iop_pinnames):
                raise RuntimeError("VOP and IOP have different set of pinnames.")

            return list(vop_pinnames)

        elif hasattr(self, "vop"):
            return list(self.vop["PinName"].unique())
        elif hasattr(self, "iop"):
            return list(self.iop["PinName"].unique())

    def get_sites(self) -> list[str]:
        if hasattr(self, "vop") and hasattr(self, "iop"):
            vop_sites = self.vop["Site"].unique()
            iop_sites = self.iop["Site"].unique()

            if set(vop_sites) != set(iop_sites):
                raise RuntimeError("VOP and IOP have different set of sites.")

            return list(vop_sites)

        elif hasattr(self, "vop"):
            return list(self.vop["Site"].unique())
        elif hasattr(self, "iop"):
            return list(self.iop["Site"].unique())

    def format_value(self, column: str, value):
        if column not in self.column_types:
            raise ValueError(f"Got unexpected column {column}")

        try:
            return self.column_types[column](value)
        except:
            return value

    def get_testsuites(self) -> list[str]:
        vop_testsuites = None
        iop_testsuites = None
        power_testsuites =None

        if hasattr(self, "vop"):
            vop_testsuites = self.vop["Testsuite"].unique().tolist()

        if hasattr(self, "iop"):
            iop_testsuites = self.iop["Testsuite"].unique().tolist()
            
        if hasattr(self, "power"):
            power_testsuites = self.power["Testsuite"].unique().tolist()


        # if vop and iop, make sure there testsuites are the same
        if (iop_testsuites is not None) and (vop_testsuites is not None):
            assert vop_testsuites == iop_testsuites
            testsuites = vop_testsuites
        elif iop_testsuites is not None:
            testsuites = iop_testsuites
        elif vop_testsuites is not None:
            testsuites = vop_testsuites
        elif power_testsuites is not None:
            testsuites = power_testsuites
        else:
            raise RuntimeError("No testsuites were found.")

        return testsuites

    @staticmethod
    def extract_group_from_full_testsuite(testsuite_full: str) -> str:
        return testsuite_full.split(".")[2]

    @staticmethod
    def float_list(list_str: list[str]) -> list[float]:
        return [float(el) for el in list_str]

    @staticmethod
    def fill_nan(arr: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(arr)
        nan_indices = np.isnan(arr)
        arr[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(valid), arr[valid])
        return arr

    def load_raw_csv(self, path) -> pd.DataFrame:
        """
        Read in the IOP or VOP data and output as dataframe.
        """
        columns = None
        num_columns = None
        content_dict = None

        with open(path, "r") as f:
            for line_num, line in enumerate(f):
                # get items in that row
                # remove ending comma
                line = line.strip()
                if line.endswith(","):
                    line = line[:-1]

                line_items = line.split(",")

                if line_num == 0:
                    # save first line as header and count number of columns
                    columns = line_items
                    num_columns = len(columns)
                    content_dict = {c: [] for c in columns}
                    print(f"Discovery {num_columns} columns where the column names are {columns}")

                else:
                    # ignore row if Sequencer column (index 7) is not "seq"
                    if line_items[7] != "seq":
                        continue

                    # add the items to the dictionary
                    # group the excess items as one group
                    last_column_removed = columns[:-1]
                    last_col = columns[-1]
                    for item, col in zip(line_items, last_column_removed):
                        content_dict[col].append(self.format_value(col, item))

                    content_dict[last_col].append(
                        self.format_value(last_col, line_items[num_columns - 1 :])
                    )

        df = pd.DataFrame(content_dict, columns=columns)
        df["row_index"] = df.index

        # sort by Seq column
        df.sort_values(by=["Seq", "row_index"], inplace=True)

        # get testsuite name after last period
        df["testsuite_last"] = df["Testsuite"].apply(lambda x: x.split(".")[-1])

        return df

    def process_vop_iop(self, voltage_or_current: str) -> pd.DataFrame:
        raw_data = None
        if voltage_or_current == "voltage":
            raw_data = self.vop
        elif voltage_or_current == "current":
            raw_data = self.iop
        elif voltage_or_current == "power":
            raw_data = self.power
        else:
            raise ValueError(f"Expected voltage or current, but got {voltage_or_current}.")

        # group by pinname and testsuite
        pinname_ts_group = raw_data.groupby(by=["Site", "PinName", "Testsuite"])

        # get plot values
        pinname_testsuite = {
            "seq": [],
            "site": [],
            "pinname": [],
            "testsuite_full": [],
            "testsuite_num": [],
            "testsuite_last": [],
            "value": [],
            "waveform": [],
            "duration": [],
            "row_index": [],
            "P95_value": [],

        }

        for (site, pinname, testsuite), group in pinname_ts_group:
            # sort group by Seq column
            group.sort_values(by=["Seq", "row_index"], inplace=True)

            # get the max voltage from the waveform
            merge_waveform = sum([w for w in group["WAVEFORM DATA"]], [])
            p95_value = np.percentile(merge_waveform, 95) if merge_waveform else None

            merge_duration = group["Duration(us)"].sum()

            pinname_testsuite["seq"].append(group["Seq"].min())
            pinname_testsuite["site"].append(site)
            pinname_testsuite["pinname"].append(pinname)
            pinname_testsuite["testsuite_full"].append(testsuite)
            pinname_testsuite["testsuite_num"].append(self.testsuites.index(testsuite))
            pinname_testsuite["testsuite_last"].append(testsuite.split(".")[-1])
            pinname_testsuite["value"].append(max(merge_waveform))
            pinname_testsuite["waveform"].append(merge_waveform)
            pinname_testsuite["duration"].append(merge_duration)
            pinname_testsuite["row_index"].append(group["row_index"].min())
            pinname_testsuite["P95_value"].append(p95_value)

             

        df = pd.DataFrame(pinname_testsuite).sort_values(by=["seq", "row_index"])
        df["testsuite_middle"] = df["testsuite_full"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 2 else x)


        return df
    def set_target_value(self, target_value: str):
        """Set the target value to be used for plotting."""
        if hasattr(self, "iop_processed") and target_value in self.iop_processed:
            self.iop_processed["target_value"] = self.iop_processed[target_value]
        if hasattr(self, "vop_processed") and target_value in self.vop_processed:
            self.vop_processed["target_value"] = self.vop_processed[target_value]




    def get_power_per_testsuite(self) -> pd.DataFrame:
        # summing testsuit across pinname
        plot_values = {
            "min_seq": [],
            "site": [],
            "testsuite_full": [],
            "testsuite_num": [],
            "testsuite_last": [],
            "max_power": [],
            "average_power": [],
            "waveform": [],
            "duration": [],
            "row_index": [],
        }

        for (site, testsuite_full), group in self.power.groupby(["site", "testsuite_full"]):
            # sum the power across rails for each timestep
            waveform = np.nansum(
                np.array([np.array(arr) for arr in group["waveform"]]), axis=0
            ).tolist()
            plot_values["min_seq"].append(group["seq"].min())
            plot_values["site"].append(site)
            plot_values["testsuite_full"].append(testsuite_full)
            plot_values["testsuite_num"].append(self.testsuites.index(testsuite_full))
            plot_values["testsuite_last"].append(testsuite_full.split(".")[-1])
            plot_values["max_power"].append(max(waveform))
            plot_values["average_power"].append(sum(waveform) / len(waveform))
            plot_values["waveform"].append(waveform)
            plot_values["duration"].append(group["duration"].unique()[0])
            plot_values["row_index"].append(group["row_index"].min())

        df = pd.DataFrame(plot_values)
        df.sort_values(by=["min_seq", "row_index"], inplace=True)
        df["testsuite_middle"] = df["testsuite_full"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 2 else x)

        return df

    def get_energy_per_testsuite(self) -> pd.DataFrame:
        # summing testsuit across pinname
        plot_values = {
            "min_seq": [],
            "site": [],
            "testsuite_full": [],
            "testsuite_num": [],
            "testsuite_last": [],
            "value": [],
            "duration": [],
            "row_index": [],
        }

        # get all waveforms of a testsuite, which is across the rails
        for (site, testsuite_full), group in self.power.groupby(["site", "testsuite_full"]):
            # sum of power waveform per rail in Watts
            # rail with all NaN is left as it-is and will be filter out in final summation
            # rail with partial NaN are filled with interpolation
            power_sum = np.array(
                [
                    (
                        np.array(arr)
                        if np.isnan(arr).all()
                        else (
                            self.fill_nan(np.array(arr)) if np.isnan(arr).any() else np.array(arr)
                        )
                    )
                    for arr in group["waveform"]
                ]
            ).sum(axis=1)

            # calculate delta time based on waveform duration and number of sample
            # assume equal time between waveform samples
            # duration given as microsecond
            duration = group["duration"].unique()[0]
            time_step = np.array([duration / len(w) * 1e-6 for w in group["waveform"]])

            # get the energy across all rail
            energy = float(np.nansum(power_sum * time_step))

            plot_values["min_seq"].append(group["seq"].min())
            plot_values["site"].append(site)
            plot_values["testsuite_full"].append(testsuite_full)
            plot_values["testsuite_num"].append(self.testsuites.index(testsuite_full))
            plot_values["testsuite_last"].append(testsuite_full.split(".")[-1])
            plot_values["value"].append(energy)
            plot_values["duration"].append(group["duration"].unique()[0])
            plot_values["row_index"].append(group["row_index"].min())

        df = pd.DataFrame(plot_values)
        df["testsuite_middle"] = df["testsuite_full"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 2 else x)

        df.sort_values(by=["min_seq", "row_index"], inplace=True)

        return df

    def find_power_violation(
        self,
        lower_threhsold: float,
        upper_threshold: float = np.inf,
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Output the testsuite that is between the lower and upper threshold,
        which is lower_threshold <= x < upper_threshold
        """
        filter_rows = self.power_per_testsuite.loc[
            (self.power_per_testsuite["average_power"] >= lower_threhsold)
            & (self.power_per_testsuite["average_power"] < upper_threshold)
        ].copy()

        # add group name column
        filter_rows["group"] = filter_rows["testsuite_full"].apply(
            lambda x: self.extract_group_from_full_testsuite(x)
        )

        # sort by group name and then power value
        filter_rows.sort_values(
            by=["group", "site", "average_power"], ascending=[True, True, False], inplace=True
        )

        # drop waveform and row_index column
        filter_rows.drop(columns=["waveform", "row_index"], inplace=True)

        # output file
        if output_file:
            filter_rows = filter_rows.reindex(
                columns=[
                    "testsuite_num",
                    "min_seq",
                    "site",
                    "testsuite_full",
                    "group",
                    "testsuite_last",
                    "average_power",
                    "duration",
                ]
            )
            filter_rows.rename(
                columns={
                    "testsuite_num": "Plot Index",
                    "min_seq": "Seq",
                    "site": "Site",
                    "testsuite_full": "Testsuite Path",
                    "group": "Testsuite Group",
                    "testsuite_last": "Testsuite Name",
                    "average_power": "Avg. Power (W)",
                    "duration": "duration (us)",
                },
                inplace=True,
            )
            filter_rows.to_csv(output_file, index=False)
            return

        return filter_rows

    def find_energy_violation(
        self,
        lower_threhsold: float,
        upper_threshold: float = np.inf,
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Output the testsuite that is between the lower and upper threshold,
        which is lower_threshold <= x < upper_threshold
        """
        filter_rows = self.energy_per_testsuite.loc[
            (self.energy_per_testsuite["value"] >= lower_threhsold)
            & (self.energy_per_testsuite["value"] < upper_threshold)
        ].copy()

        # add group name column
        filter_rows["group"] = filter_rows["testsuite_full"].apply(
            lambda x: self.extract_group_from_full_testsuite(x)
        )

        # sort by group name and then power value
        filter_rows.sort_values(
            by=["group", "site", "value"], ascending=[True, True, False], inplace=True
        )

        # drop row_index column
        filter_rows.drop(columns=["row_index"], inplace=True)

        # output file
        if output_file:
            filter_rows = filter_rows.reindex(
                columns=[
                    "testsuite_num",
                    "min_seq",
                    "site",
                    "testsuite_full",
                    "group",
                    "testsuite_last",
                    "value",
                    "duration",
                ]
            )
            filter_rows.rename(
                columns={
                    "testsuite_num": "Plot Index",
                    "min_seq": "Seq",
                    "site": "Site",
                    "testsuite_full": "Testsuite Path",
                    "group": "Testsuite Group",
                    "testsuite_last": "Testsuite Name",
                    "value": "Energy (J)",
                    "duration": "duration (us)",
                },
                inplace=True,
            )
            filter_rows.to_csv(output_file, index=False)
            return

        return filter_rows

    def plot_voltage(self, output_file: str = None) -> None:
        """
        Plot the rail by testsuite where the data is max voltage of each testsuite-rail
        """

        print("Plotting VOP.")
        voltage_groups = self.vop_processed.groupby("site")
        for site, group in voltage_groups:
            fig = px.scatter(
                group.drop(columns=["waveform"]),
                x="testsuite_middle",
                y="value",
                color="pinname",
                hover_data={
                    "site": True,
                    "pinname": True,
                    "testsuite_last": True,
                    "value": True,
                    "duration": True,
                    "testsuite_full": True,
                    "testsuite_num": True,
                },
                labels={"testsuite_middle": "Testflow", "value": "Max Voltage (V)"},
            )

            fig.update_layout(
                title="Max Voltage for Rail & Testflow",
                title_font=dict(size=24, color="darkblue", family="Arial"),
                title_x=0.5,  # Center the title
                width=1200,  # Width in pixels
                height=600,  # Height in pixels
            )

            if output_file:
                # save plot as html
                # create parent directory if not exists
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                site_str = str(site) if len(voltage_groups) > 1 else ""
                outfile_renamed = output_file.replace(".html", f"_{site_str}.html")
                fig.write_html(outfile_renamed)
            else:
                fig.show()
                
    def plot_current(self, output_file: str = None) -> None:
        """
        Plot the rail by testsuite where the data is max current of each testsuite-rail
        """

        print("Plotting IOP.")
        current_groups = self.iop_processed.groupby("site")
        for site, group in current_groups:
            fig = px.scatter(
                group.drop(columns=["waveform"]),
                x="testsuite_middle",
                y="value",
                color="pinname",
                hover_data={
                    "site": True,
                    "pinname": True,
                    "testsuite_last": True,
                    "value": True,
                    "duration": True,
                    "testsuite_full": True,
                    "testsuite_num": True,
                },
                labels={"testsuite_middle": "Testflow", "value": "Max Current (A)"},
            )

            fig.update_layout(
                title="Max Current for Rail & Testflow",
                title_font=dict(size=24, color="darkblue", family="Arial"),
                title_x=0.5,  # Center the title
                width=1200,  # Width in pixels
                height=600,  # Height in pixels
            )

            if output_file:
                # save plot as html
                # create parent directory if not exists
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                site_str = str(site) if len(current_groups) > 1 else ""
                outfile_renamed = output_file.replace(".html", f"_{site_str}.html")
                fig.write_html(outfile_renamed)
            else:
                fig.show()

    def plot_power(self, thresholds: list[float] = None, output_file: str = None, per_pin: bool = False) -> None:
        if thresholds is None:
            thresholds = POWER_THRESHOLDS
        if per_pin:
            print("Plotting Power per Pin.")
        power_groups = self.power.groupby("site")
        for site, group in power_groups:
            group = group.copy()
            group["value"] = group["waveform"].apply(lambda w: max(w) if isinstance(w, list) else None)

            fig = px.scatter(
                group.drop(columns=["waveform"]),
                x="testsuite_middle",
                y="value",
                color="pinname",
                hover_data={
                    "site": True,
                    "pinname": True,
                    "testsuite_last": True,
                    "value": True,
                    "duration": True,
                    "testsuite_full": True,
                    "testsuite_num": True,
                },
                labels={"testsuite_middle": "Testflow", "value": "Max Power (W)"},
            )

            fig.update_layout(
                title="Max Power for Rail & Testflow",
                title_font=dict(size=24, color="darkblue", family="Arial"),
                title_x=0.5,
                width=1200,
                height=600,
            )

            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                site_str = str(site) if len(power_groups) > 1 else ""
                outfile_renamed = output_file.replace(".html", f"_{site_str}.html")
                fig.write_html(outfile_renamed)
            else:
                fig.show()
                return

    # Default: per-testsuite power plot
        print("Plotting Power per Testsuite.")
        exceeding_count = {}
        num_ts = len(self.power_per_testsuite["testsuite_full"].unique())
        
        for threshold in thresholds:
            exceeding_df = self.find_power_violation(lower_threhsold=threshold)
            count = len(exceeding_df["testsuite_full"].unique())
        exceeding_count[threshold] = count
        exceeding_count = dict(sorted(exceeding_count.items(), key=lambda item: item[0]))

        title = f"Power for Each Testsuite (Sum across {len(self.pinnames)} Rails)<br>Total Testsuites: {num_ts}"
        if len(exceeding_count) > 0:
            for threshold, count in exceeding_count.items():
                title += f" | above {threshold}W: {count}"

        min_threshold = min(thresholds)
        self.power_per_testsuite["avg_exceed"] = self.power_per_testsuite["average_power"].apply(
        lambda x: ("rgb(255, 170, 187)" if x >= min_threshold else "rgb(99, 110, 250)")
    )
        self.power_per_testsuite["max_exceed"] = self.power_per_testsuite["max_power"].apply(
        lambda x: "rgb(255, 50, 50)" if x >= min_threshold else "rgb(50, 50, 255)"
    )

        fig = go.Figure()
        for site, group in self.power_per_testsuite.groupby("site"):
            fig.add_trace(
            go.Scatter(
                x=group["testsuite_num"],
                y=group["average_power"],
                customdata=group[
                    ["testsuite_last", "average_power", "duration", "testsuite_full"]
                ],
                hovertemplate="testsuite_num=%{x}<br>"
                + "testsuite=%{customdata[0]}<br>"
                + "avg power=%{customdata[1]}<br>"
                + "duration(us)=%{customdata[2]}<br>"
                + "full testsuite=%{customdata[3]}<br>",
                name=f"Average Power Site {site}",
                mode="markers",
                marker=dict(color=group["avg_exceed"]),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=group["testsuite_num"],
                y=group["max_power"],
                customdata=group[
                    ["testsuite_last", "max_power", "duration", "testsuite_full"]
                ],
                hovertemplate="testsuite_num=%{x}<br>"
                + "testsuite=%{customdata[0]}<br>"
                + "max power=%{customdata[1]}<br>"
                + "duration(us)=%{customdata[2]}<br>"
                + "full testsuite=%{customdata[3]}<br>",
                name=f"Max Power Site {site}",
                mode="markers",
                marker=dict(color=group["max_exceed"]),
            )
        )

        fig.update_layout(
        xaxis_title="Testsuite",
        yaxis_title="Power (W)",
        title=title,
        title_font=dict(size=22, color="darkblue", family="Arial"),
        title_x=0.5,
        width=1200,
        height=500,
    )

        for threshold in thresholds:
            fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=threshold,
            y1=threshold,
            line=dict(color="red", width=2),
            xref="paper",
            yref="y",
        )
        fig.add_annotation(
            x=-100,
            y=threshold + 1,
            text=f"{threshold}W",
            showarrow=False,
            font=dict(size=16, color="red", family="Arial"),
        )

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_file)
        else:
            fig.show()

        self.power_per_testsuite.drop(columns=["max_exceed", "avg_exceed"], inplace=True)
   
        # === Unified Power Plot with Thresholds and Pin Names ===
        print("Generating unified power plot with thresholds...")

        if thresholds is None:
            thresholds = POWER_THRESHOLDS

        # Count threshold violations
        exceeding_count = {}
        num_ts = len(self.power_per_testsuite["testsuite_full"].unique())
        for threshold in thresholds:
            exceeding_df = self.find_power_violation(lower_threhsold=threshold)
            count = len(exceeding_df["testsuite_full"].unique())
            exceeding_count[threshold] = count
        exceeding_count = dict(sorted(exceeding_count.items(), key=lambda item: item[0]))

        # Title
        title = f"Unified Power Plot (Sum across {len(self.pinnames)} Rails)<br>Total Testsuites: {num_ts}"
        if len(exceeding_count) > 0:
            for threshold, count in exceeding_count.items():
                title += f" | above {threshold}W: {count}"

        # Color coding for threshold exceedance
        min_threshold = min(thresholds)
        self.power_per_testsuite["avg_exceed"] = self.power_per_testsuite["average_power"].apply(
            lambda x: "rgb(255, 170, 187)" if x >= min_threshold else "rgb(99, 110, 250)"
        )
        self.power_per_testsuite["max_exceed"] = self.power_per_testsuite["max_power"].apply(
            lambda x: "rgb(255, 50, 50)" if x >= min_threshold else "rgb(50, 50, 255)"
        )

        fig = go.Figure()

        # Per-pin power (max per rail)
        for (site, pinname), group in self.power.groupby(["site", "pinname"]):
            group = group.copy()
            group["value"] = group["waveform"].apply(lambda w: max(w) if isinstance(w, list) else None)
            fig.add_trace(
                go.Scatter(
                    x=group["testsuite_num"],
                    y=group["value"],
                    mode="markers",
                    name=f"{pinname} (Site {site})",
                    marker=dict(symbol="circle", size=6),
                    customdata=group[["testsuite_last", "value"]],
                    hovertemplate="Pin: " + pinname + "<br>Test: %{customdata[0]}<br>Max Power: %{customdata[1]} W"
                )
            )


        # Per-testsuite average and max power
        for site, group in self.power_per_testsuite.groupby("site"):
            fig.add_trace(
                go.Scatter(
                    x=group["testsuite_num"],
                    y=group["average_power"],
                    mode="markers",
                    name=f"Avg Power (Site {site})",
                    marker=dict(symbol="diamond", color=group["avg_exceed"], size=10),
                    customdata=group[["testsuite_last", "average_power", "duration", "testsuite_full"]],
                    hovertemplate="Test: %{customdata[0]}<br>Avg Power: %{customdata[1]} W<br>Duration: %{customdata[2]} us<br>Full Path: %{customdata[3]}"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=group["testsuite_num"],
                    y=group["max_power"],
                    mode="markers",
                    name=f"Max Power (Site {site})",
                    marker=dict(symbol="square", color=group["max_exceed"], size=10),
                    customdata=group[["testsuite_last", "max_power", "duration", "testsuite_full"]],
                    hovertemplate="Test: %{customdata[0]}<br>Max Power: %{customdata[1]} W<br>Duration: %{customdata[2]} us<br>Full Path: %{customdata[3]}"
                )
            )

        # Layout and threshold lines
        fig.update_layout(
            title=title,
            xaxis_title="Testsuite Number",
            yaxis_title="Power (W)",
            title_font=dict(size=22, color="darkblue", family="Arial"),
            title_x=0.5,
            width=1200,
            height=600
        )

        for threshold in thresholds:
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=threshold,
                y1=threshold,
                line=dict(color="red", width=2),
                xref="paper",
                yref="y",
            )
            fig.add_annotation(
                x=-100,
                y=threshold + 1,
                text=f"{threshold}W",
                showarrow=False,
                font=dict(size=16, color="red", family="Arial"),
            )

        # Save unified plot
        if output_file:
            unified_output = output_file.replace(".html", "_unified.html")
            Path(unified_output).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(unified_output)
        else:
            fig.show()

        # Clean up
        self.power_per_testsuite.drop(columns=["max_exceed", "avg_exceed"], inplace=True)

 



    def plot_energy(self, thresholds: list[float] = None, output_file: str = None) -> None:
        """
        Plot the energy (= sum [sum_rail of [(duration/num_samples) (power_i)]]) of each testsuite
        """
        if thresholds is None:
            thresholds =ENERGY_THRESHOLDS
        # get total number of testsuits
        num_ts = len(self.energy_per_testsuite["testsuite_full"].unique())

        # get number of testsuites above each threshold
        exceeding_count = {}
        for threshold in thresholds:
            exceeding_df = self.find_energy_violation(lower_threhsold=threshold)
            count = len(exceeding_df["testsuite_full"].unique())
            exceeding_count[threshold] = count
        exceeding_count = dict(sorted(exceeding_count.items(), key=lambda item: item[0]))

        # create title
        title = f"Energy Consumption for Each Testsuite (Sum across {len(self.pinnames)} Rails)<br>Total Testsuites: {num_ts}"
        if len(exceeding_count) > 0:
            for threshold, count in exceeding_count.items():
                title += f" | above {threshold}J: {count}"

        # create color for energy data points that surpasses minimal threshold
        min_threshold = min(thresholds)
        self.energy_per_testsuite["energy_exceed"] = self.energy_per_testsuite["value"].apply(
            lambda x: ("rgb(255, 170, 187)" if x >= min_threshold else "rgb(99, 110, 250)")
        )

        print("Plotting Energy.")
        fig = go.Figure()

        for site, group in self.energy_per_testsuite.groupby("site"):
            fig.add_trace(
                go.Scatter(
                    x=group["testsuite_num"],
                    y=group["value"],
                    customdata=group[["testsuite_last", "value", "duration", "testsuite_full"]],
                    hovertemplate="testsuite_num=%{x}<br>"
                    + "testsuite=%{customdata[0]}<br>"
                    + "Energy=%{customdata[1]}<br>"
                    + "duration(us)=%{customdata[2]}<br>"
                    + "full testsuite=%{customdata[3]}<br>",
                    name=f"Site {site}",
                    mode="markers",
                    marker=dict(color=group["energy_exceed"]),
                )
            )

        fig.update_layout(xaxis_title="Testsuite", yaxis_title="Energy (J)")

        fig.update_layout(
            title=title,
            title_font=dict(size=22, color="darkblue", family="Arial"),
            title_x=0.5,  # Center the title,
            width=1200,  # Width in pixels
            height=500,  # Height in pixels
        )

        # draw threshold lines
        for threshold in thresholds:
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=threshold,
                y1=threshold,
                line=dict(color="red", width=2),
                xref="paper",
                yref="y",
            )

            fig.add_annotation(
                x=-100,  # x position of the text
                y=threshold + 0.1,  # y position of the text
                text=f"{threshold}J",  # text to display
                showarrow=False,  # whether to show an arrow pointing to the text
                font=dict(size=16, color="red", family="Arial"),
            )

        if output_file:
            # save plot as html
            # create parent directory if not exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_file)
        else:
            fig.show()
            
    # def generate_avg_power_summary(self, output_file: str = "avg_power_summary.html") -> None:
    #     print("Generating average power summary per pin and testsuite.")
        
    #     # Ensure power data is processed
    #     if not hasattr(self, "power"):
    #         raise RuntimeError("Power data not loaded.")

    #     # Compute average power per pin and testsuite
    #     summary_df = self.power.copy()
    #     summary_df["avg_power"] = summary_df["waveform"].apply(
    #         lambda w: sum(w) / len(w) if isinstance(w, list) and len(w) > 0 else None
    #     )

    #     grouped = summary_df.groupby(["pinname", "testsuite_full"])["avg_power"].mean().reset_index()
    #     grouped.rename(columns={"testsuite_full": "Testsuite Path", "avg_power": "Avg Power (W)"}, inplace=True)

    #     # Save to HTML
    #     grouped.to_html(output_file, index=False, float_format="%.6f")

    #     print(f"Average power summary saved to {output_file}")
    def generate_avg_power_summary(self, output_file: str = "avg_power_summary.html") -> None:
        print("Generating detailed average power summary per pin and testsuite.")

        if not hasattr(self, "power"):
            raise RuntimeError("Power data not loaded.")

        summary_df = self.power.copy()

        # Compute average power
        summary_df["Avg Power (W)"] = summary_df["waveform"].apply(
            lambda w: sum(w) / len(w) if isinstance(w, list) and len(w) > 0 else None
        )

        # Extract testsuite group and name
        summary_df["Testsuite Group"] = summary_df["testsuite_full"].apply(lambda x: x.split(".")[2] if len(x.split(".")) > 2 else "")
        summary_df["Testsuite Name"] = summary_df["testsuite_full"].apply(lambda x: x.split(".")[-1])

        # Rename for clarity
        summary_df.rename(columns={
            "pinname": "Pin Name",
            "seq": "Seq Index",
            "site": "Site",
            "testsuite_full": "Testsuite Path",
            "duration": "Duration (us)"
        }, inplace=True)

       

        # Select and reorder columns
        final_df = summary_df[[
            "Pin Name", "Seq Index", "Site", "Testsuite Path",
            "Testsuite Group", "Testsuite Name", "Avg Power (W)", "Duration (us)",
        ]]

        # Save to HTML
        #final_df.to_html(output_file, index=False, float_format="%.6f")
        #save to csv
        final_df.to_csv(output_file, index=False, float_format="%.6f")

        print(f"Detailed average power summary saved to {output_file}")



   

   
    
def generate_rail_summary_html_max_only(instances, output_file: str = "rail_summary_max_only.html") -> None:
    

    print("Generating simplified rail summary (max only) for multiple instances.")

    def summarize_max(df, label):
        if label == "Power":
            df = df.copy()
            df["value"] = df["waveform"].apply(lambda x: max(x) if isinstance(x, list) and len(x) > 0 else np.nan)
        grouped = df.groupby("pinname")["value"].max().reset_index()
        grouped.columns = ["PinName", f"{label}_max"]
        return grouped

    all_summaries = []

    for idx, instance in enumerate(instances):
        voltage_max = summarize_max(instance.vop_processed, "Voltage")
        current_max = summarize_max(instance.iop_processed, "Current")
        power_max = summarize_max(instance.power, "Power")

        summary_df = voltage_max.merge(current_max, on="PinName", how="outer")
        summary_df = summary_df.merge(power_max, on="PinName", how="outer")
        summary_df.insert(0, "Device", f"Device_{idx+1}")

        all_summaries.append(summary_df)

    final_df = pd.concat(all_summaries, ignore_index=True)

    # Save to HTML
    final_df.to_html(output_file, index=False)
    print(f"Simplified rail summary saved to {output_file}")
    

    
 
    
            


# %%
def main():
    hpsv_s = [] 
    hpsv = HighPowerSpecViolation(VOP_PATH, IOP_PATH, POWER_PATH)
    hpsv.plot_voltage(VOLTAGE_PLOT_OUTPUT)
    hpsv.plot_current(CURRENT_PLOT_OUTPUT)
    hpsv.plot_power(POWER_THRESHOLDS, POWER_PLOT_OUTPUT)
    hpsv.plot_energy(ENERGY_THRESHOLDS, ENERGY_PLOT_OUTPUT)
    hpsv.set_target_value("P95_value")
    hpsv.plot_current("p95_current_plot.html")
    # P95 Voltage
    hpsv.set_target_value("p95_value")
    hpsv.plot_voltage("p95_voltage_plot.html")

    # P95 Power
    hpsv.set_target_value("p95_value")
    hpsv.plot_power(output_file="p95_power_plot.html", per_pin=True)


    hpsv.find_power_violation(min(POWER_THRESHOLDS), output_file=POWER_OUTPUT_CSV)
    hpsv.find_energy_violation(min(ENERGY_THRESHOLDS), output_file=ENERGY_OUTPUT_CSV)
    hpsv_s.append(hpsv)
        
    generate_rail_summary_html_max_only(hpsv_s, "rail_summary.html")
   # hpsv.generate_avg_power_summary("avg_power_summary.html")
    hpsv.generate_avg_power_summary("avg_power_summary.csv")


    

if __name__ == "__main__":
    main()

