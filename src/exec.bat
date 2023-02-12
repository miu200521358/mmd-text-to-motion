set TEXT="”Ş‚ÍU‚è•Ô‚Á‚Äè‚ğU‚Á‚½"
set SECONDS=6

python executor.py --parent-dir O:/MMD_Auto_Trace/text --text %TEXT% --seed 0 --seconds %SECONDS% --num_repetitions 2 --process text2move,mix,motion --verbose 20 --log-mode 1 --lang ja
