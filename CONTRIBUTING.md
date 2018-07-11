# Contributing to AMLA

## Feature enhancements
If you have made an improvement to an existing feature, please send us a pull request.
See the section on [pull requests](#pull-requests). 

## Feature requests
Feature requests are welcome. 

* If you would like to see a new major feature added, please consider making a feature request
Current feature requests are listed in the (issue tracker)(#issuetracker).
If you do not see an existing feature request that covers your feature, please add a new feature request to the issue tracker. 
Please use the label "Feature Request" on the issue when adding a new feature request.

* Feature requests with major codebase changes need a proposal document. 
Minor feature requests don't need an associated proposal.
If your feature request already exists in the tracker, there may be already be a proposal submitted for it.
Current proposals are organised by area [here](./docs/proposals)
The proposals follow the naming convention: docs/proposals/<area>/<github username>-<proposal name>.md, where the proposal name is an area specific name: E.g. docs/proposals/deployer/pkamath-kubeflow.md
If your feature request fits in an existing area, and a proposal exists, please consider contributing to 
the proposal

* If a proposal does not exist, please consider authoring a proposal. 
Proposals should be reasonably well written, but do not need to be perfect.
Cover at least the following:
    * Definition: What the feature is
    * Problem/Motivation: Why the feature is needed
    * Solution: What the feature will do
    * Design: How the feature will be implemented, why is this the chosen method, other possible ways to implement
    * Plan: When the feature development will start and how long it will take

[Here](./docs/proposals/scaleout/pkamath-paralleltrain.md) is an example of a proposal.
Once your proposal is in reasonable shape, send us a pull request for the proposal and update the issue with your proposal.
Proposals will be discussed by the community. 
If there are multiple proposals in a single area, the proposals will reviewed and, depending on feedback, may be merged into a single final proposal:  proposals/<area>/proposal-final-<proposal-name>.md
Implementation will begin once there is agreement within the community on the proposal.

## Issue tracker

The [issue tracker](https://github.com/amla/issues) is the preferred channel for [bug reports](#bug-reports), [features requests](#feature-requests) and [submitting pull requests](#pull-requests)

* Please **do not** use the issue tracker for personal support requests.

* Use [GitHub's "reactions" feature](https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments)
  Please **do not** post comments consisting solely of "+1" or ":thumbsup:".

## Bug reports

A bug is a _demonstrable problem_ that is caused by the code in the repository.
Good bug reports are extremely helpful, so thanks!

Guidelines for bug reports:

0. **Validate and lint your code** &mdash; to ensure your problem isn't caused by a simple error in your own code.

1. **Use the GitHub issue search** &mdash; check if the issue has already been reported.

2. **Check if the issue has been fixed** &mdash; try to reproduce it using the latest `master` or development branch in the repository.

3. **Isolate the problem** &mdash; ideally create a [reduced test case](https://css-tricks.com/reduced-test-cases/).

A good bug report shouldn't leave others needing to chase you up for more
information. Please try to be as detailed as possible in your report. What is
your environment? What steps will reproduce the issue? All these details will help people to fix
any potential bugs.

Example:

> Short and descriptive example bug report title
>
> A summary of the issue and the environment in which it occurs.
> Include the steps required to reproduce the bug.
>
> 1. This is the first step
> 2. This is the second step
> 3. Further steps, etc.
>
> `<url>` - a link to the reduced test case
>
> Any other information you want to share that is relevant to the issue being
> reported. This might include the lines of code that you have identified as
> causing the bug, and potential solutions (and your opinions on their
> merits).

## Pull requests

Good pull requests—patches, improvements, new features—are a fantastic
help. They should remain focused in scope and avoid containing unrelated
commits.

**Please ask first** before embarking on any significant pull request (e.g.
implementing features, refactoring code, porting to a different language),
otherwise you risk spending a lot of time working on something that the
project's developers might not want to merge into the project.

Adhering to the following process is the best way to get your work
included in the project:

1. [Fork](https://help.github.com/fork-a-repo/) the project, clone your fork,
   and configure the remotes:

   ```bash
   # Clone your fork of the repo into the current directory
   git clone https://github.com/<your-username>/amla.git
   # Navigate to the newly cloned directory
   cd amla
   # Assign the original repo to a remote called "upstream"
   git remote add upstream https://github.com/ciscoai/amla.git
   ```

2. If you cloned a while ago, get the latest changes from upstream:

   ```bash
   git checkout master
   git pull upstream master
   ```

3. Create a new topic branch (off the main project development branch) to
   contain your feature, change, or fix:

   ```bash
   git checkout -b <topic-branch-name>
   ```

4. Commit your changes in logical chunks. Use Git's
   [interactive rebase](https://help.github.com/articles/interactive-rebase)
   feature to tidy up your commits before making them public.

5. Locally merge (or rebase) the upstream development branch into your topic branch:

   ```bash
   git pull [--rebase] upstream master
   ```

6. Push your topic branch up to your fork:

   ```bash
   git push origin <topic-branch-name>
   ```

7. [Open a Pull Request](https://help.github.com/articles/using-pull-requests/) with a clear title and description against the `master` branch.

**IMPORTANT**: By submitting a patch, you agree to allow the project owners to license your work under the terms of the [Apache 2.0 License](LICENSE).

### Semantic Git commit messages

Inspired by Sparkbox's awesome article on [semantic commit messages](http://seesparkbox.com/foundry/semantic_commit_messages). Please use following commit message format.

* chore (updating build tasks etc; no production code change) -> ```git test -m 'chore: commit-message-here'```
* docs (changes to documentation) -> ```git commit -m 'docs: commit-message-here'```
* feat (new feature) -> ```git commit -m 'feat: commit-message-here'```
* fix (bug fix) -> ```git commit -m 'fix: commit-message-here'```
* refactor (refactoring production code) -> ```git commit -m 'refactor: commit-message-here'```
* style (formatting, missing semi colons, etc; no code change) -> ```git commit -m 'style: commit-message-here'```
* test (adding missing tests, refactoring tests; no production code change) -> ```git test -m 'refactor: commit-message-here'```

## Code guidelines

### Python

[Adhere to the PEP8 standard.](https://www.python.org/dev/peps/pep-0008/)

- Use pylint/autopep8 to check your code for coding standards. A score of is 8/10 and higher is considered acceptable.

## License

By contributing your code, you agree to license your contribution under the [Apache 2.0 License](LICENSE).

## Acknowledgements

Parts of this document have been adapted from [CoreUI](https://github.com/coreui/coreui)'s contribution guidelines.
