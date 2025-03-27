import os
import re
import pandas as pd
from typing import Dict, Any
from collections import defaultdict

def parse_review_dates(review_text: str) -> Dict[str, str]:
    """Extract creation and update dates from review text."""
    dates_match = re.search(r'dates=ReviewDates\[creation=(\d{4}-\d{2}-\d{2}), update=(\d{4}-\d{2}-\d{2})\]', review_text)
    if dates_match:
        return {
            'creation_date': dates_match.group(1),
            'update_date': dates_match.group(2)
        }
    return {'creation_date': None, 'update_date': None}

def parse_self_line(line: str) -> Dict[str, str]:
    """Parse the @SELF line into components."""
    parts = line.split('\t')
    if len(parts) >= 4:
        return {
            'name': parts[0],
            'url': parts[1],
            'provider': parts[2],
            'provider_url': parts[3]
        }
    return {}

def parse_attribute_line(line: str) -> Dict[str, Any]:
    """Parse an attribute line into key-value pair."""
    try:
        # Remove @ATTRIBUTE prefix
        line = line.replace('@ATTRIBUTE', '').strip()
        
        # Split into name and value
        parts = line.split(':', 1)
        if len(parts) != 2:
            # Try to extract any useful information even from malformed lines
            if '(' in line and ')' in line:
                # Try to extract type and value from parentheses
                match = re.search(r'\((.*?),\s*(.*?)\)', line)
                if match:
                    type_name, value = match.groups()
                    return {type_name.strip(): value.strip()}
            return {}
            
        name, value = parts
        name = name.strip()
        value = value.strip()
        
        # Handle different value types
        if value.startswith('('):
            # Handle tuple format (TYPE, VALUE)
            value = value.strip('()')
            type_name, actual_value = value.split(',', 1)
            actual_value = actual_value.strip()
            
            # Handle different value types
            if type_name in ['LAST_UPDATE', 'RELEASE_DATE']:
                # Keep dates as strings
                return {type_name.strip(): actual_value}
            elif type_name in ['LAYOUT', 'TYPE', 'PROVIDER', 'THEME', 'FEATURES', 'OTHER_TAGS', 'TECHNOLOGY', 'OBJECTS', 'GENRE']:
                # Keep these as strings or lists
                if actual_value.startswith('[') and actual_value.endswith(']'):
                    actual_value = eval(actual_value)
                return {type_name.strip(): actual_value}
            else:
                # Try to convert to float for numeric values
                try:
                    if actual_value.replace('.', '').replace('-', '').isdigit():
                        actual_value = float(actual_value)
                except ValueError:
                    pass  # Keep as string if conversion fails
            
            return {type_name.strip(): actual_value}
        return {name: value}
    except Exception as e:
        print(f"Warning: Error parsing attribute line '{line}': {str(e)}")
        return {}

def parse_related_slot(line: str) -> Dict[str, str]:
    """Parse a related slot line into components."""
    parts = line.split('\t')
    if len(parts) >= 4:
        return {
            'name': parts[0],
            'url': parts[1],
            'provider': parts[2],
            'provider_url': parts[3]
        }
    return {}

def parse_slot_file(file_path: str) -> Dict[str, Any]:
    """Parse a single slot file into structured data."""
    data = {
        'self': {},
        'review': {},
        'attributes': {},
        'related_slots': [],
        'top_provider_slots': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                
                if line.startswith('@SELF'):
                    if i + 1 < len(lines):
                        data['self'] = parse_self_line(lines[i + 1].strip())
                        i += 2
                    else:
                        i += 1
                elif line.startswith('@REVIEW'):
                    if i + 1 < len(lines):
                        review_text = lines[i + 1].strip()
                        data['review'] = {
                            'text': review_text,
                            **parse_review_dates(review_text)
                        }
                        i += 2
                    else:
                        i += 1
                elif line.startswith('@ATTRIBUTE'):
                    if i + 1 < len(lines):
                        attr_data = parse_attribute_line(lines[i + 1].strip())
                        if attr_data:
                            data['attributes'].update(attr_data)
                        i += 2
                    else:
                        i += 1
                elif line.startswith('@RELATED_SLOT'):
                    if i + 1 < len(lines):
                        slot_data = parse_related_slot(lines[i + 1].strip())
                        if slot_data:
                            data['related_slots'].append(slot_data)
                        i += 2
                    else:
                        i += 1
                elif line.startswith('@TOP_PROVIDER_SLOT'):
                    if i + 1 < len(lines):
                        slot_data = parse_related_slot(lines[i + 1].strip())
                        if slot_data:
                            data['top_provider_slots'].append(slot_data)
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None
    
    return data

def process_all_slots(directory: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all slot files in the directory and return DataFrames for slots and providers."""
    all_data = []
    provider_stats = defaultdict(lambda: {
        'game_count': 0,
        'total_rtp': 0,
        'avg_rtp': 0,
        'total_max_win': 0,
        'avg_max_win': 0,
        'game_types': set(),
        'features': set(),
        'themes': set(),
        'technologies': set()
    })
    
    # Get absolute path
    abs_dir = os.path.abspath(directory)
    
    # List all files
    files = [f for f in os.listdir(abs_dir) if f.endswith('.slot')]
    
    # Create dictionaries for mapping names to IDs
    slot_id_map = {}
    provider_id_map = {}
    current_slot_id = 1
    current_provider_id = 1
    
    # First pass: collect all slot names and create IDs
    for filename in files:
        file_path = os.path.join(abs_dir, filename)
        slot_data = parse_slot_file(file_path)
        
        if slot_data and slot_data['self']:
            slot_name = slot_data['self'].get('name')
            if slot_name and slot_name not in slot_id_map:
                slot_id_map[slot_name] = current_slot_id
                current_slot_id += 1
            
            # Also collect related and top provider slot names
            for slot in slot_data['related_slots']:
                slot_name = slot.get('name')
                if slot_name and slot_name not in slot_id_map:
                    slot_id_map[slot_name] = current_slot_id
                    current_slot_id += 1
            
            for slot in slot_data['top_provider_slots']:
                slot_name = slot.get('name')
                if slot_name and slot_name not in slot_id_map:
                    slot_id_map[slot_name] = current_slot_id
                    current_slot_id += 1
    
    # Second pass: process all data with complete ID mapping
    for filename in files:
        file_path = os.path.join(abs_dir, filename)
        slot_data = parse_slot_file(file_path)
        
        if slot_data and slot_data['self']:  # Only process if we have valid data
            # Flatten the data structure
            flat_data = {}
            
            # Add self data and slot ID
            slot_name = slot_data['self'].get('name')
            flat_data['slot_id'] = slot_id_map.get(slot_name)
            flat_data.update(slot_data['self'])
            
            # Add review data
            if slot_data['review']:
                flat_data.update({
                    'review_text': slot_data['review'].get('text'),
                    'creation_date': slot_data['review'].get('creation_date'),
                    'update_date': slot_data['review'].get('update_date')
                })
            
            # Add attributes with default values
            flat_data.update({
                'RTP': slot_data['attributes'].get('RTP', -99.0),
                'MAX_WIN_RELATIVE': slot_data['attributes'].get('MAX_WIN_RELATIVE', -99.0),
                'MIN_BET': slot_data['attributes'].get('MIN_BET', -99.0),
                'MAX_BET': slot_data['attributes'].get('MAX_BET', -99.0),
                'VARIANCE': slot_data['attributes'].get('VARIANCE', -99),
                'HIT_FREQUENCY': slot_data['attributes'].get('HIT_FREQUENCY', -99.0),
                'GAME_SIZE': slot_data['attributes'].get('GAME_SIZE', -99.0),
                'TYPE': slot_data['attributes'].get('TYPE', 'Unknown'),
                'LAYOUT': slot_data['attributes'].get('LAYOUT', 'Unknown'),
                'FEATURES': slot_data['attributes'].get('FEATURES', []),
                'THEME': slot_data['attributes'].get('THEME', []),
                'TECHNOLOGY': slot_data['attributes'].get('TECHNOLOGY', []),
                'OTHER_TAGS': slot_data['attributes'].get('OTHER_TAGS', []),
                'OBJECTS': slot_data['attributes'].get('OBJECTS', []),
                'GENRE': slot_data['attributes'].get('GENRE', []),
                'LAST_UPDATE': slot_data['attributes'].get('LAST_UPDATE', None),
                'RELEASE_DATE': slot_data['attributes'].get('RELEASE_DATE', None)
            })
            
            # Process related slots and top provider slots to only store IDs
            related_slot_ids = []
            for slot in slot_data['related_slots']:
                slot_name = slot.get('name')
                if slot_name in slot_id_map:
                    related_slot_ids.append(slot_id_map[slot_name])
            
            top_provider_slot_ids = []
            for slot in slot_data['top_provider_slots']:
                slot_name = slot.get('name')
                if slot_name in slot_id_map:
                    top_provider_slot_ids.append(slot_id_map[slot_name])
            
            # Add related slots and top provider slots as ID lists
            flat_data.update({
                'related_slot_ids': related_slot_ids,
                'top_provider_slot_ids': top_provider_slot_ids,
                'related_slots_count': len(related_slot_ids),
                'top_provider_slots_count': len(top_provider_slot_ids)
            })
            
            all_data.append(flat_data)
            
            # Update provider statistics
            provider = slot_data['self'].get('provider')
            if provider:
                if provider not in provider_id_map:
                    provider_id_map[provider] = current_provider_id
                    current_provider_id += 1
                provider_stats[provider]['provider_id'] = provider_id_map[provider]
                provider_stats[provider]['game_count'] += 1
                if flat_data['RTP'] != -99.0:
                    provider_stats[provider]['total_rtp'] += flat_data['RTP']
                if flat_data['MAX_WIN_RELATIVE'] != -99.0:
                    provider_stats[provider]['total_max_win'] += flat_data['MAX_WIN_RELATIVE']
                if flat_data['TYPE'] != 'Unknown':
                    provider_stats[provider]['game_types'].add(flat_data['TYPE'])
                if isinstance(flat_data['FEATURES'], list):
                    provider_stats[provider]['features'].update(flat_data['FEATURES'])
                if isinstance(flat_data['THEME'], list):
                    provider_stats[provider]['themes'].update(flat_data['THEME'])
                if isinstance(flat_data['TECHNOLOGY'], list):
                    provider_stats[provider]['technologies'].update(flat_data['TECHNOLOGY'])
    
    # Calculate averages and convert sets to lists for provider stats
    for provider in provider_stats:
        stats = provider_stats[provider]
        if stats['game_count'] > 0:
            stats['avg_rtp'] = stats['total_rtp'] / stats['game_count']
            stats['avg_max_win'] = stats['total_max_win'] / stats['game_count']
        stats['game_types'] = list(stats['game_types'])
        stats['features'] = list(stats['features'])
        stats['themes'] = list(stats['themes'])
        stats['technologies'] = list(stats['technologies'])
    
    # Create provider DataFrame with proper index
    provider_df = pd.DataFrame.from_dict(provider_stats, orient='index')
    provider_df.index.name = 'provider_name'
    
    return pd.DataFrame(all_data), provider_df

def print_statistics(df: pd.DataFrame, provider_df: pd.DataFrame):
    """Print basic statistics about the dataset."""
    print("\nBasic Statistics:")
    print(f"Total number of games: {len(df)}")
    print(f"Total number of unique slots: {df['slot_id'].nunique()}")
    print(f"Total number of unique providers: {provider_df.index.nunique()}")
    
    # Provider statistics
    if 'provider' in df.columns:
        print("\nTop 10 providers by game count:")
        for provider, row in provider_df.nlargest(10, 'game_count').iterrows():
            print(f"- {provider}: {row['game_count']} games")
            print(f"  Provider ID: {row['provider_id']}")
            print(f"  Average RTP: {row['avg_rtp']:.2f}%")
            print(f"  Average Max Win: {row['avg_max_win']:.2f}x")
            print(f"  Game Types: {', '.join(row['game_types'])}")
    
    # RTP statistics
    if 'RTP' in df.columns:
        rtp_stats = df[df['RTP'] != -99.0]['RTP'].describe()
        print(f"\nRTP Statistics (excluding -99.0):")
        print(f"- Mean: {rtp_stats['mean']:.2f}%")
        print(f"- Median: {rtp_stats['50%']:.2f}%")
        print(f"- Min: {rtp_stats['min']:.2f}%")
        print(f"- Max: {rtp_stats['max']:.2f}%")
    
    # Max win statistics
    if 'MAX_WIN_RELATIVE' in df.columns:
        win_stats = df[df['MAX_WIN_RELATIVE'] != -99.0]['MAX_WIN_RELATIVE'].describe()
        print(f"\nMax Win Statistics (x bet, excluding -99.0):")
        print(f"- Mean: {win_stats['mean']:.2f}x")
        print(f"- Median: {win_stats['50%']:.2f}x")
        print(f"- Min: {win_stats['min']:.2f}x")
        print(f"- Max: {win_stats['max']:.2f}x")
    
    # Feature statistics
    if 'FEATURES' in df.columns:
        all_features = []
        for features in df['FEATURES'].dropna():
            if isinstance(features, list):
                all_features.extend(features)
        feature_counts = pd.Series(all_features).value_counts()
        print("\nTop 10 most common features:")
        for feature, count in feature_counts.head(10).items():
            print(f"- {feature}: {count} games")
    
    # Related slots statistics
    if 'related_slots' in df.columns:
        related_counts = df['related_slots_count'].describe()
        print("\nRelated Slots Statistics:")
        print(f"- Mean: {related_counts['mean']:.2f}")
        print(f"- Median: {related_counts['50%']:.2f}")
        print(f"- Min: {related_counts['min']:.2f}")
        print(f"- Max: {related_counts['max']:.2f}")
    
    # Top provider slots statistics
    if 'top_provider_slots' in df.columns:
        top_counts = df['top_provider_slots_count'].describe()
        print("\nTop Provider Slots Statistics:")
        print(f"- Mean: {top_counts['mean']:.2f}")
        print(f"- Median: {top_counts['50%']:.2f}")
        print(f"- Min: {top_counts['min']:.2f}")
        print(f"- Max: {top_counts['max']:.2f}")

if __name__ == "__main__":
    try:
        # Process all slot files
        df, provider_df = process_all_slots('raw_game_data')
        
        # Save to CSV
        df.to_csv('slot_data.csv', index=False)
        provider_df.to_csv('provider_stats.csv')
        print(f"Successfully processed {len(df)} slot files and saved to slot_data.csv and provider_stats.csv")
        
        # Print statistics
        print_statistics(df, provider_df)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 