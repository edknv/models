# Contributing to Merlin Models

If you are interested in contributing to Merlin Models, your contributions will fall
into three categories:

1. You want to report a bug, feature request, or documentation issue:
   - File an [issue](https://github.com/NVIDIA-Merlin/models/issues/new/choose)
     and describe what you encountered or what you want to see changed.
   - The NVIDIA-Merlin team evaluates the issues and triages them, scheduling
     them for a release. If you believe the issue needs priority attention,
     comment on the issue to notify the team.
2. You want to propose a new feature and implement it:
   - Post about your intended feature to discuss the design and
     implementation with the NVIDIA-Merlin team.
   - Once we agree that the plan looks good, go ahead and implement it, using
     the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue:
   - Follow the [code contributions](#code-contributions) guide below.
   - If you need more context on a particular issue, please ask and the
     NVIDIA-Merlin team will provide the context.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA-Merlin/models/blob/main/README.md)
   to learn how to setup the development environment.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/NVIDIA-Merlin/models/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   or [help wanted](https://github.com/NVIDIA-Merlin/models/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.
3. Comment on the issue to say that you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/NVIDIA-Merlin/models/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. After your pull request is reviewed and approved, an NVIDIA-Merlin team member merges it.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

After you have your feet wet and are comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/NVIDIA-Merlin/models/projects).

> **Pro Tip:** Always look at the release board with the highest number for
> issues to work on. This is where the NVIDIA-Merlin developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know that you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Label your PRs

This repository uses the release-drafter action to draft and create our change log.

Please add one of the following labels to your PR to specify the type of contribution
and help categorize the PR in our change log:

- `breaking` -- The PR creates a breaking change to the API.
- `bug` -- The PR fixes a problem with the code.
- `feature` or `enhancement` -- The PR introduces a backward-compatible feature.
- `documentation` or `examples` -- The PR is an addition or update to documentation.
- `build`, `dependencies`, `chore`, or `ci` -- The PR is related to maintaining the
  repository or the project.

By default, an unlabeled PR is listed at the top of the change log and is not
grouped under a heading like _Features_ that groups similar PRs.
Labeling the PRs so we can categorize them is preferred.

If, for some reason, you do not believe your PR should be included in the change
log, you can add the `skip-changelog` label.
This label excludes the PR from the change log.

For more information, see `.github/release-drafter.yml` in the repository
or go to <https://github.com/release-drafter/release-drafter>.

## Adding classes, methods, and so on

If you add a class, method, or function, make an update to `docs/source/api.rst`
to reference your change. If you remove one of these items, please also remove
it from the same `api.rst` file.

> In this second case, you probably want to mark your PR with the `breaking`
> label when you open the PR.

Please make sure to build the documentation by following the guidance in
`docs/README.md` and check that you do not introduce any build errors or warnings.

The following line is an example of a warning. Check that the specified class,
`CachedCrossBatchSampler` is still part of the software or remove it from
`api.rst` and label your PR with the `breaking` label.

```terminal
WARNING: [autosummary] failed to import 'merlin.models.tf.CachedCrossBatchSampler': no module named merlin.models.tf.CachedCrossBatchSampler
```

Also look for warnings that report mistakes with formatting docstrings:

```terminal
merlin/models/merlin/models/tf/blocks/retrieval/base.py:docstring of merlin.models.tf.blocks.retrieval.base.DualEncoderBlock.__ini
t__:11: WARNING: Unexpected indentation.
```

**TIP**: To add a link from a docstring to class documentation in the
Merlin Core repository, specify the fully qualified class name and prefix it
with a tilde like the following example:

```python
Parameters
----------
schema : ~merlin.schema.Schema
    The `Schema` with the input features
deep_block: Block
    Block (structure) of deep model.
```

The [Intersphinx](https://docs.readthedocs.io/en/stable/guides/intersphinx.html)
extension truncates the text to [Schema](https://nvidia-merlin.github.io/core/main/api/merlin.schema.html)
and makes it a link.

## Attribution

Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
