
### To package for the spin glass server:

Edit the rsync line in `package_for_spinglassserver.sh` to have tars copied to correct local machine (for browser-based upload)

``` bash 
cd RND_J
bash ../package_for_spinglassserver.sh
cd ..
```

Upload the produced tars to the Spin Glass Server.  Wait for responses. 
Place responses in files. To parse the emails to gs_energy files:

``` bash
cd  RND_J
python parse_email.py -f 'email_file.txt'
cd ..
```

