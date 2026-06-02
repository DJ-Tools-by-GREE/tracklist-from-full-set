Writing outputs …
Reading Engine DJ hotcues …
  Tracklist     : /Users/janmuller/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4/tracklist.csv
  Heatmap       : /Users/janmuller/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4/confidence_heatmap.png
  Transitions   : /Users/janmuller/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4/transitions.html
Traceback (most recent call last):
  File "/Users/janmuller/Git Repos/tracklist-from-full-set/tracklist_generator.py", line 2484, in <module>
    main()
  File "/Users/janmuller/Git Repos/tracklist-from-full-set/tracklist_generator.py", line 2480, in main
    write_review_ui(tracks, set_duration, OUTPUT_REVIEW_UI)
  File "/Users/janmuller/Git Repos/tracklist-from-full-set/tracklist_generator.py", line 2309, in write_review_ui
    html = html.replace("__PAYLOAD__", json.dumps(payload))
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/__init__.py", line 231, in dumps
    return _default_encoder.encode(obj)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/encoder.py", line 199, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/encoder.py", line 257, in iterencode
    return _iterencode(o, 0)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool is not JSON serializable