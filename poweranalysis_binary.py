import struct
import argparse
from settings import BINARY_PROF_PATH

import json
import numpy as np

 
 
import pandas as pd
import plotly.express as px


 


def parse_header(f):
    header = b""
    while True:
        byte = f.read(1)
        if byte == b":":
            break
        header += byte

    if (len(header) > 100):
        print (f"Error at {f.tell()} bytes")
    parts = header.decode('utf-8').split('/')
    site = int(parts[0])
    signal = parts[1]
    vc = parts[2]
    timestamp = int(parts[3])
    length = int(parts[4])
    samplerate= float(parts[5])
    return site, signal, vc, timestamp, length, samplerate

def read_file(filename):  
    data = []
    execution_info = {}

    with open(filename, 'rb') as f:
        while True:
            start_byte = f.read(1)
            if not start_byte:
                break
            if start_byte == b":":
                site, signal, vc, timestamp, length, samplerate = parse_header(f)
                # f.read(length * 8)
                MAX_LENGTH = 10_000_000  # Set a reasonable upper limit

                if length > MAX_LENGTH:
                    raise MemoryError(f"Length too large: {length}")

                # chunk_size = 100000
                # doubles = []
                # remaining = length
                # while remaining > 0:
                #     read_len = min(chunk_size, remaining)
                #     chunk = struct.unpack(f'>{read_len}d', f.read(read_len * 8))
                #     doubles.extend(chunk)
                #     remaining -= read_len
                 

                chunk_size = 100000
                doubles = np.empty(length, dtype='>f8')  # Preallocate memory for all doubles
                offset = 0
                remaining = length

                while remaining > 0:
                    read_len = min(chunk_size, remaining)
                    chunk = struct.unpack(f'>{read_len}d', f.read(read_len * 8))
                    doubles[offset:offset + read_len] = chunk
                    offset += read_len
                    remaining -= read_len


                data.append((site, signal, vc, timestamp, length, samplerate, doubles))
            elif start_byte == b"{":
                f.seek(-1, 1)
                json_bytes = f.read()
                json_string = json_bytes.decode('utf-8')
                execution_info = json.loads(json_string)
            else:
                print (f"Error at {f.tell()} bytes")
                 
                #return 
                return [],{}
                 
    return data, execution_info



 


 
def detect_regions(waveform, mask=2):
    regions = []
    in_region = False
    start = 0
    for i, value in enumerate(waveform):
        raw = struct.unpack('>Q', struct.pack('>d', value))[0]
        in_opseq = (raw & mask) != 0
        if in_opseq and not in_region:
            start = i
            in_region = True
        elif not in_opseq and in_region:
            regions.append((start, i))
            in_region = False
    return regions

 


 
 









def main():
  
    

    data, execution_info = read_file(BINARY_PROF_PATH)

    sites = sorted(set(entry[0] for entry in data))  # Get unique site IDs
    for site in sites:
        site_data = [entry for entry in data if entry[0] == site]
        plot_voltage_directly(site_data, execution_info, site)
        plot_current_directly(site_data, execution_info, site)
       
        plot_power_directly(site_data, execution_info, site)
         
        generate_rail_summary_html_max_only(site_data, execution_info, output_file=f"rail_summary_site{site}.html")



    df = pd.DataFrame(data, columns=["Site", "PinName", "VC", "Timestamp", "Length", "SampleRate", "Waveform"])

    
    signal_map = {}
    
    for entry in data:
        site, signal, vc, timestamp, length, samplerate, waveform = entry
        signal_map.setdefault((site, signal), {})[vc] = waveform
        
    testsuite_map = {}
    for entry in data:
        site, signal, vc, timestamp, length, samplerate, waveform = entry
        if str(site) in execution_info:
            entries = execution_info[str(site)]
            closest = min(entries, key=lambda e: abs(e["siteUnspecificInfo"]["startTimestamp"] - timestamp))
            matched_fqn = closest["siteUnspecificInfo"]["fullyQualifiedName"]
            testsuite_map[(site, signal, vc)] = matched_fqn
    df["Testsuite"] = df.apply(lambda row: testsuite_map.get((row["Site"], row["PinName"], row["VC"])), axis=1)


def plot_voltage_directly(data, execution_info, site):

    print(" Plotting voltage data...")  
    voltage_entries = [entry for entry in data if entry[2] == 'V']
    rows = []

    for site, signal, vc, timestamp, length, samplerate, waveform in voltage_entries:
        # Find closest matching testsuite
        matched_fqn = ""
        if str(site) in execution_info:
            entries = execution_info[str(site)]
            closest = min(entries, key=lambda e: abs(e["siteUnspecificInfo"]["startTimestamp"] - timestamp))
            matched_fqn = closest["siteUnspecificInfo"]["fullyQualifiedName"]

        rows.append({
            "Site": site,
            "PinName": signal,
            "Testsuite": matched_fqn,
            "testsuite_last": matched_fqn.split(".")[-1],
            "value": max(waveform), # the max value is simply the highest value 
            "p95": np.percentile(waveform, 95),
            "Duration(us)": length,
        })

    df = pd.DataFrame(rows)
    df["testsuite_middle"] = df["Testsuite"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 2 else x)


    fig = px.scatter(
        df,
        x="testsuite_middle",
        y="value",
        color="PinName",
       # hover_data=["Site", "PinName", "testsuite_last", "value", "Duration(us)", "Testsuite"],
       hover_data=["Site", "PinName", "testsuite_last", "value", "p95", "Duration(us)", "Testsuite"],

        labels={"testsuite_middle": "Test Flow", "value": "Max Voltage (V)"},
        #title="Max Voltage for Rail & Test Flow"
        title="Max Voltage and P95 for Rail & Test Flow"



    )
    #fig.update_layout(title_x=0.5, width=1200, height=600)
    fig.update_layout(
        title_x=0.5,
        width=1400,  # wider plot
        height=1000,  # taller plot to fit legend
        legend=dict(
            orientation="v",  # vertical layout
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # move legend outside plot area
            font=dict(size=10)  # optional: shrink font if needed
        )
    )

    fig.show()
    fig.write_html(f"Voltage_plotprofz_site{site}.html")

    print("Voltage plot displayed and saved as 'voltage_plotprofz.html'.")

def plot_current_directly(data, execution_info, site):

    print(" Plotting current data...")

    current_entries = [entry for entry in data if entry[2] == 'C']
    rows = []

    for site, signal, vc, timestamp, length, samplerate, waveform in current_entries:
        matched_fqn = ""
        if str(site) in execution_info:
            entries = execution_info[str(site)]
            closest = min(entries, key=lambda e: abs(e["siteUnspecificInfo"]["startTimestamp"] - timestamp))
            matched_fqn = closest["siteUnspecificInfo"]["fullyQualifiedName"]

        rows.append({
            "Site": site,
            "PinName": signal,
            "Testsuite": matched_fqn,
            "testsuite_last": matched_fqn.split(".")[-1],
            "value": max(waveform),  # the max value is simply the highest value 
            "p95": np.percentile(waveform, 95),
            "Duration(us)": length,
        })

    df = pd.DataFrame(rows)
    df["testsuite_middle"] = df["Testsuite"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 2 else x)


    fig = px.scatter(
        df,
        x="testsuite_middle",
        y="value",
        color="PinName",
        #hover_data=["Site", "PinName", "testsuite_last", "value", "Duration(us)", "Testsuite"],
        hover_data=["Site", "PinName", "testsuite_last", "value", "p95", "Duration(us)", "Testsuite"],

        labels={"testsuite_middle": "Test Flow", "value": "Max Current (A)"},
        #title="Max Current for Rail & Test Flow"
        title="Max Current and P95 for Rail & Test Flow"



    )
    #fig.update_layout(title_x=0.5, width=1200, height=600)
    fig.update_layout(
        title_x=0.5,
        width=1400,  # wider plot
        height=1000,  # taller plot to fit legend
        legend=dict(
            orientation="v",  # vertical layout
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # move legend outside plot area
            font=dict(size=10)  # optional: shrink font if needed
        )
    )

    fig.show()
    fig.write_html(f"Current_plotprofz_site{site}.html")


    print("Current plot displayed and saved as 'Current_plotprofz.html'.")
 
def plot_power_directly(data, execution_info, site):
    print(f" Plotting power data for site {site}...")
    from collections import defaultdict
    grouped = defaultdict(dict)
    for entry in data:
        site_id, pinname, vc, timestamp, length, samplerate, waveform = entry
        if str(site_id) in execution_info:
            entries = execution_info[str(site_id)]
            closest = min(entries, key=lambda e: abs(e["siteUnspecificInfo"]["startTimestamp"] - timestamp))
            testsuite = closest["siteUnspecificInfo"]["fullyQualifiedName"]
            grouped[(site_id, pinname, testsuite)][vc] = waveform

    power_data = []
    for (site_id, pinname, testsuite), vc_map in grouped.items():
        if 'V' in vc_map and 'C' in vc_map:
            voltage = vc_map['V']
            current = vc_map['C']
            power = [v * i for v, i in zip(voltage, current)]
            power_data.append({
                "Site": site_id,
                "PinName": pinname,
                "Testsuite": testsuite,
                "testsuite_last": testsuite.split(".")[-1],
                "value": max(power),
                "p95": np.percentile(power, 95),
                "Duration(us)": len(power),
            })

    df = pd.DataFrame(power_data)
    df["testsuite_middle"] = df["Testsuite"].apply(lambda x: x.split(".")[1] if len(x.split(".")) > 2 else x)
    fig = px.scatter(
        df,
        x="testsuite_middle",
        y="value",
        color="PinName",
        hover_data=["Site", "PinName", "testsuite_last", "value", "p95", "Duration(us)", "Testsuite"],
        labels={"testsuite_middle": "Test Flow", "value": "Max Power (W)"},
        title="Max Power and P95 for Rail & Test Flow"
    )
    #fig.update_layout(title_x=0.5, width=1200, height=600)
    fig.update_layout(
        title_x=0.5,
        width=1400,  # wider plot
        height=1000,  # taller plot to fit legend
        legend=dict(
            orientation="v",  # vertical layout
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # move legend outside plot area
            font=dict(size=10)  # optional: shrink font if needed
        )
    )

    fig.show()
    fig.write_html(f"Power_plotprofz_site{site}.html")
    print(f"Power plot saved as 'Power_plotprofz_site{site}.html'.")




 
 
def generate_rail_summary_html_max_only(data, execution_info, output_file="rail_summary_prof.html"):
    print("Generating rail summary with max and P95 values...")

    def summarize(entries, label):
        rows = []
        for site, signal, vc, timestamp, length, samplerate, waveform in entries:
            if str(site) in execution_info:
                entries_info = execution_info[str(site)]
                closest = min(entries_info, key=lambda e: abs(e["siteUnspecificInfo"]["startTimestamp"] - timestamp))
                testsuite = closest["siteUnspecificInfo"]["fullyQualifiedName"]
                rows.append({
                    "PinName": signal,
                    "Testsuite": testsuite,
                   # f"{label}_max": max(waveform) if waveform else None,
                    f"{label}_max": np.max(waveform) if waveform is not None else None,

                    #f"{label}_p95": np.percentile(waveform, 95) if waveform else None
                    f"{label}_p95": np.percentile(waveform, 95) if waveform is not None else None

                })
        return pd.DataFrame(rows)

    voltage_df = summarize([e for e in data if e[2] == 'V'], "Voltage")
    current_df = summarize([e for e in data if e[2] == 'C'], "Current")

    # Compute power
    from collections import defaultdict
    grouped = defaultdict(dict)
    power_rows = []
    for entry in data:
        site, signal, vc, timestamp, length, samplerate, waveform = entry
        grouped[(site, signal, timestamp)][vc] = (waveform, site)

    for (site, signal, timestamp), vc_map in grouped.items():
        if 'V' in vc_map and 'C' in vc_map:
            voltage, site_v = vc_map['V']
            current, site_c = vc_map['C']
            power = [v * i for v, i in zip(voltage, current)]
            if str(site) in execution_info:
                entries_info = execution_info[str(site)]
                closest = min(entries_info, key=lambda e: abs(e["siteUnspecificInfo"]["startTimestamp"] - timestamp))
                testsuite = closest["siteUnspecificInfo"]["fullyQualifiedName"]
                power_rows.append({
                "PinName": signal,
                "Testsuite": testsuite,
                "Power_max": max(power) if power else None,
                #"Power_p95": f"{np.percentile(power, 95):.6f}" if power else None
                "Power_p95": np.percentile(power, 95) if power else None

            })


    power_df = pd.DataFrame(power_rows)

    
    summary_df = pd.concat([voltage_df, current_df, power_df], axis=0)
    summary_df = summary_df.groupby("PinName").agg({
        "Voltage_max": "max",
        "Voltage_p95": "max",
        "Current_max": "max",
        "Current_p95": "max",
        "Power_max": "max",
        "Power_p95": "max"
    }).reset_index()
  
    summary_df.to_html(output_file, index=False)
    print(f"Rail summary with max and P95 saved to {output_file}")




    


    
   

if __name__ == '__main__':

    main()
