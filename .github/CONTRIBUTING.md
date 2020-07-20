
1. Please sign the [UCX contributors agreement](http://www.openucx.org/license).

1. Please follow the [code style](https://github.com/openucx/ucx/blob/master/docs/CodeStyle.md)
   and [logging style](https://github.com/openucx/ucx/blob/master/docs/LoggingStyle.md).

1. Make sure automatic tests pass.

1. Request a review by [mentioning](https://github.com/blog/821-mention-somebody-they-re-notified)
   or [requesting a review](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/requesting-a-pull-request-review)
   from the relevant reviewer.

1. PR which is reviewed and currently waiting for a fix and/or response, will be
   marked with "Waiting for Author Response" by the reviewer,

1. In order to address review comments, please push a new commit with the necessary changes.
   1. Do not rebase, squash, amend, or force-push while review is still in progress (comments 
      were already posted and the pull request was not approved yet). Doing so will discard 
      the previous version of the code for which comments were already posted.  
   1. If there is a merge conflict with master branch, use merge commit to resolve the conflict 
      (do not rebase)  
   1. If force-push was done by accident, please rebase the code on the last commit for which comments
      were posted and force-push again. This will restore the previous commits.   

1. After getting an approval from one or more of UCX maintainers the PR can be merged.  
   At this point, the author can squash all commits into one and force-push, to keep a
   clean git history. It's recommended to not rebase on master (unless required because 
   of a merge conflict) so github could show that no files were changed.

More details [here](http://github.com/openucx/ucx/wiki/Guidance-for-contributors).
