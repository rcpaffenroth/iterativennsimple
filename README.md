# iterativenn_simple
A simple version of iterativenn for anyone to use

## Assumptions

This README assumed a basic familiarity with github.com, vscode, and codespaces.  In particular, it is *not* intended to be a tutorial on using standard tools like Python, Linux, git, etc.  

However, to get you started here are some links:

- A YouTube video that shows how to use codespaces on github: https://www.youtube.com/watch?v=ozuDPmcC1io&list=PLmsFUfdnGr3wTl-NCblzcrEv2lFSX975-&index=1
- The vscode documentation: https://code.visualstudio.com/docs
- The codespaces documentation: https://docs.github.com/en/codespaces/guides

## "Big ideas"

The idea of this resository is to provide a simple version of iterativenn that anyone can use.  It is not intended to be a full-featured version of iterativenn.  It is intended to be a starting point for anyone who wants to use iterativenn.

It is also intended as a way to get started with codespaces.  Codespaces is a way to run vscode in the cloud.  It is a great way to get started with Python, etc.  It is also a great way to get started with iterativenn.

Now, it is not the intention that your development will always be in the cloud.  It is expected that you will eventually want to run your code on your own computer (which is way more complicated, but definitely worth doing :-).  But, codespaces is a great way to get started.

When you are ready for next steps then you can contact me at rcpaffenroth@wpi.edu

## How to use

Open a terminal and run the following command to install pytorch, etc.

```bash
scripts/post-create.sh
```

You will only need to this once per rebuild.  Then, you can click on

```
notebooks/1-rcp-generate-data.ipynb
```
to get started.

## Tips and tricks

Don't forget to set your Python environment.  The one used here is

```
. venv/bin/python
```


