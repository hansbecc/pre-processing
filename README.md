# Project for Pre-processing data
Methods for data pre-processing

## git commands
To clone project in your PC **the first time** to start working

```
git clone https://github.com/hansbecc/pre-processing.git
```

Configure your account in your PC just **the first time**

```
git config --global user.email "you@example.com"
```

```
git config --global user.name "Your Name"
```

**If** an other developer **wrote or edited the project** in the remote master (in the github master)  for updateing your local files everytime before working in your PC.

```
git pull origin master
```
### Common git commands after work in your local PC files for uploading
**Preparing Step:** To add file or folder to the enviroment (in your **local PC**) for sending to the github master project (Internet) by the `git commit` command

```
git add file.xx
```

**Preparing Step:** To join all files added with a message text included about the changes (e.g. "Update the text README")

```
git commit -m "write the changes done here (start with infinitive verb: update, add,etc)"
```

**Sending Step:** Merge your changes with the master project in the github (fromyour PC to the Internet)

```
git push origin master
```

### Other git commands useful
To see the status of the files in your actual enviroment (e.g. files added, removed,etc)

```
git status
```