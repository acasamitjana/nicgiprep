from typing import Optional


class LogBIDSLoader(object):
    """Validate  BIDS dataset queries.

    Attributes
    ----------
    num_files : int
        Default expected number of files per query.
    """

    def __init__(self, num_files: int) -> None:
        """
        Parameters
        ----------
        num_files : int
            Default expected number of files per query.
        """
        self.num_files = num_files

    def check_length(
        self, file_list: list[str], curr_len: Optional[int] = None
    ) -> dict:
        """Check that the number of found files matches the expected count.

        Parameters
        ----------
        file_list : list of str
            File paths returned by a BIDS query.
        curr_len : int, optional
            Expected number of files. Defaults to ``self.num_files``.

        Returns
        -------
        dict
            Result dictionary with keys:

            - ``'exit_code'`` : int — ``0`` on success, ``-1`` on mismatch.
            - ``'file'`` : str, list, or None — matched file(s), or ``None``
              on failure.
            - ``'log'`` : str — human-readable message (empty on success).
        """
        curr_len = self.num_files if curr_len is None else curr_len

        if len(file_list) != curr_len:
            return {
                "exit_code": -1,
                "file": None,
                "log": "Files found: "
                + str(len(file_list))
                + "; files expected: "
                + str(curr_len)
                + ". Please refine the search.\n",
            }

        else:
            d = {"exit_code": 0, "log": ""}
            if curr_len == 1:
                return {**d, "file": file_list[0]}
            else:
                return {**d, "file": file_list}
