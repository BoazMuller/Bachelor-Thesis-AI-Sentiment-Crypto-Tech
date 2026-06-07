import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# Set paths
DATA_PATH = Path("results/tables/dfm_em/dfm_em_dataset.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def plot_ais_minimalist():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.set_index('date')
    
    # We will plot the raw_ais series
    series = df['raw_ais']
    
    # 2. Create the plot with a minimalist aesthetic
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot the main line
    ax.plot(series.index, series, color='#2F6DB3', linewidth=2, zorder=2)
    
    # Fill between the line and zero to give it a bit of weight
    ax.fill_between(series.index, series, 0, color='#E9F2FF', alpha=0.5, zorder=1)
    ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--', zorder=1)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')
    
    ax.tick_params(colors='#333333', which='both')
    ax.set_ylabel('AI Sentiment Index (Raw AIS)', color='#333333', fontsize=11, weight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=0, ha='center')
    
    # 3. Annotate major events corresponding to peaks and troughs
    # List of dictionaries specifying dates, labels, and offset placements to avoid overlap.
    events = [
        {'date': '2024-04-04', 'text': 'Q2 2024 AI Momentum', 'xytext': (30, -150)},
        {'date': '2024-05-14', 'text': 'GPT-4o & Google I/O', 'xytext': (40, 30)},
        {'date': '2024-09-12', 'text': 'OpenAI o1 Release', 'xytext': (-25, 40)},
        {'date': '2024-10-09', 'text': 'AI Nobel Prizes', 'xytext': (25, 55)},
        {'date': '2024-11-30', 'text': 'DeepSeek V3', 'xytext': (0, -45)},
        {'date': '2025-02-13', 'text': 'EU AI Act Passed', 'xytext': (0, -65)},
        {'date': '2025-09-04', 'text': 'ROI Reality Check', 'xytext': (-30, -30)},
        {'date': '2025-10-01', 'text': "Claude Hype", 'xytext': (30, 70)}
    ]
    
    for event in events:
        date_obj = pd.to_datetime(event['date'])
        
        # Fallback to nearest date if exact date is not in index (e.g. weekends)
        if date_obj not in series.index:
            closest_idx = series.index.get_indexer([date_obj], method='nearest')[0]
            date_obj = series.index[closest_idx]
            
        y_val = series.loc[date_obj]
        
        # Draw the annotation with an arrow
        ax.annotate(
            event['text'],
            xy=(date_obj, y_val),
            xytext=event.get('xytext', (0, 30)),
            textcoords='offset points',
            ha='center',
            fontsize=9.5,
            weight='bold',
            color='#111827',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F8FAFC', ec='#CBD5E1', lw=1),
            arrowprops=dict(arrowstyle="->", color='#6B7280', connectionstyle="arc3,rad=0.1", lw=1.2),
            zorder=3
        )
        
    ax.set_title("AI Sentiment Over Time", loc='left', fontsize=14, weight='bold', pad=15, color='#111827')
    
    # 4. Save figure
    out_file = OUT_DIR / "figure_ais_timeseries.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Minimalist AIS plot saved to: {out_file}")

if __name__ == "__main__":
    plot_ais_minimalist()
