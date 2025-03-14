
import os 

def load_poems(root_dir):
    """
    Load poems from a directory with 'forms' and 'topics' main folders
    Returns list of dicts with poem info
    """
    poems_data = []
    
    # Iterate through main categories (forms and topics)
    for main_category in ['forms', 'topics']:
        main_path = os.path.join(root_dir, main_category)
        if not os.path.exists(main_path):
            print(f"Warning: {main_path} not found")
            continue
            
        # Iterate through subfolders (abc, love, etc.)
        for sub_category in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub_category)
            if os.path.isdir(sub_path):
                # Read each poem file in the subfolder
                for filename in os.listdir(sub_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(sub_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                poem_text = f.read().strip()
                                if len(poem_text.split()) >= 5:  # Filter very short texts
                                    poems_data.append({
                                        'text': poem_text,
                                        'main_category': main_category,
                                        'sub_category': sub_category,
                                        'title': filename.replace('.txt', '')
                                    })
                        except Exception as e:
                            print(f"Error reading {file_path}: {str(e)}")
    
    print(f"Loaded {len(poems_data)} poems:")
    print(f"- Forms: {sum(1 for p in poems_data if p['main_category'] == 'forms')}")
    print(f"- Topics: {sum(1 for p in poems_data if p['main_category'] == 'topics')}")
    return poems_data