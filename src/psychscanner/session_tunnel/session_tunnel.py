from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger


class SessionTunnel:
    """A class to manage session tunnels for logging and tracking.

    This class handles the creation, management, and retrieval of session logs
    for a specific project. It provides methods to create directories, serialize
    logs, and retrieve past session data.

    Attributes:
    ----------
    curr_path : Path
        The current working directory.
    tunnel_status : str
        The status of the tunnel.
    project_name : str
        The name of the project.
    tunnel_dir : Path
        The directory where tunnel logs are stored.
    tunnel_file : Path
        The file where tunnel logs are written.
    tunnel_data : None or dict
        Data related to the tunnel (default is None).
    session : logger
        The logger instance for the session.
    run_type : None or str
        The type of the current run (default is None).
    session_id : str
        The ID of the current session (default is "--").
    state : None or str
        The state of the current session (default is None).
    """
    def __init__(self, tunnel_status: str,project_name:str,tunnel_dir:Path) -> None:
        """Initialize the SessionTunnel with tunnel status and project name.

        Parameters:
        ----------
        tunnel_status : str
            The status of the tunnel.
        project_name : str
            The name of the project.
        """
        self.curr_path = Path.cwd()
        self.tunnel_status = tunnel_status

        self.tunnel_dir = tunnel_dir
        self.tunnel_file = self.tunnel_dir / f"{project_name}-tunnel.json"

        self.tunnel_data = None
        self.session = logger
        self.run_type = None
        self.session_id = "--"
        self.state = None

    def tunnel_exists(self) -> bool:
        """Check if the tunnel log file exists.

        Returns:
        -------
        bool
            True if the tunnel log file exists, False otherwise.
        """
        return self.tunnel_file.exists()

    def create_tunnel_dir(self) -> None:
        """Create the tunnel directory if it does not already exist.

        This method checks whether the tunnel directory and file exist.
        If they do not exist, it creates the directory with the necessary
        parent directories.
        """
        if not self.tunnel_exists():
            self.tunnel_dir.mkdir(parents=True, exist_ok=True)

    def serialize(self, record: object) -> str:
        """Serialize a log record into a JSON string.

        Parameters:
        ----------
        record : object
            The log record to be serialized.

        Returns:
        -------
        str
            A JSON string representation of the log record.
        """
        subset = {
            "timestamp": record["time"].timestamp(),
            "level": record["level"].name,
            "run_type": self.run_type,
            "session_id": self.session_id,
            "state": self.state,
            "message": record["message"],
        }
        return json.dumps(subset)

    def patching(self, record: object) -> None:
        """Patch a log record with serialized data.

        Parameters:
        ----------
        record : dict
            The log record to be patched.
        """
        record["extra"]["serialized"] = self.serialize(record)

    def create_tunnel(self, tunnel_file: Path | None = None) -> None:
        """Create a log file for the session tunnel.

        Parameters:
        ----------
        tunnel_file : Path or None, optional
            The path to the log file. If None, the default tunnel file path is used.
        """
        if tunnel_file is None:
            tunnel_file = self.tunnel_file

        self.session = self.session.patch(self.patching)
        self.session.add(
            tunnel_file,
            format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {extra[id]} | {extra[run_type]}"
            "| {extra[state]} | {message}",
            serialize=True,
            level="TRACE",
        )
        self.session_id ="BEGIN"
        self.run_type = "BEGIN"
        self.state = "BEGIN"

        session_over = []  # Initialize with a default value
        if self.tunnel_exists():
            tunnel_past = self.load_tunnel_logs()
            scan_on_off_events = [i for i in tunnel_past if i["level"] == "CRITICAL"]
            session_over = [i for i in scan_on_off_events if i["message"]=="END"]
            #session_over = True if len(session_over) == 1 else False

        if len(session_over)==1:
            msg = "Session already has ended. Delete old files to run."
            raise ValueError(msg)

        with self.session.contextualize(
            id=self.session_id, run_type=self.run_type, state=self.state
        ):
            self.session.critical("BEGIN")

    # def get_past_session(self, tunnel_file=None):
    # pass
    def end_checkpoint(self):
        self.session_id = "END"
        self.run_type = "END"
        self.state = "END"
        with self.session.contextualize(
            id=self.session_id, run_type=self.run_type, state=self.state
        ):
            self.session.critical("END")


    def scan_checkpoint(self, session_id: object, run_type: str = ">>>>SCAN<<<<", state: object = "--") -> None:
        """Log a checkpoint for a scan session.

        Parameters:
        ----------
        session_id : str
            The ID of the scan session.
        run_type : str, optional
            The type of the run, default is ">>>>SCAN<<<<".
        state : object, optional
            The state of the scan session, default is "--".
        """
        self.run_type = run_type
        self.session_id = session_id
        self.state = state
        with self.session.contextualize(
            id=self.session_id, run_type=self.run_type, state=self.state
        ):
            self.session.info("scan-checkpoint")


    def subscan_checkpoint(self, subscan_id: object, run_type: str = ">>SUB_SCAN<<", state: str = "--") -> None:
        """Log a checkpoint for a subscan session.

        Parameters:
        ----------
        subscan_id : str
            The ID of the subscan session.
        run_type : str, optional
            The type of the run, default is ">>SUB_SCAN<<".
        state : str, optional
            The state of the subscan session, default is "--".
        """
        self.run_type = run_type
        self.session_id = subscan_id
        self.state = state
        with self.session.contextualize(id=self.session_id,run_type=self.run_type, state=self.state):
            self.session.trace("sub-scan-checkpoint")

    def load_tunnel_logs(self, tunnel_file: Path | None = None, *, return_all: bool = False, to_frame: bool = False) -> pd.DataFrame | list:
        """Load tunnel logs from the specified file.

        Parameters:
        ----------
        tunnel_file : Path or None, optional
            The path to the log file. If None, the default tunnel file path is used.
        return_all : bool, optional
            Whether to return all logs or only serialized logs. Default is False.
        to_frame : bool, optional
            Whether to convert logs to a pandas DataFrame. Default is True.

        Returns:
        -------
        pd.DataFrame or list
            A DataFrame or list containing the logs, depending on the parameters.
        """
        if tunnel_file is None:
            tunnel_file = self.tunnel_file
        all_logs = []
        logs_serialized = []
        with tunnel_file.open(encoding="utf-8") as file:
            for _i, line in enumerate(file.readlines()):
                all_logs.append(json.loads(line))
                logs_serialized.append(
                    json.loads(all_logs[-1]["record"]["extra"]["serialized"])
                )

        if to_frame:
            all_logs = pd.DataFrame(all_logs)
            logs_serialized = pd.DataFrame(logs_serialized)

        if return_all:
            return all_logs

        return logs_serialized

    def all_past_scans(self, past_scan: pd.DataFrame | None = None) -> pd.DataFrame:
        """Retrieve all past scan sessions.

        Parameters:
        ----------
        past_scan : pd.DataFrame or None, optional
            DataFrame containing past scan sessions. If None, it retrieves all past scans.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing all past scan sessions.
        """
        if past_scan is None:
            past_scan = self.load_tunnel_logs()

        return past_scan.loc[past_scan["level"] == "INFO"]


    def all_past_sscans(self, past_scan: pd.DataFrame | None = None) -> pd.DataFrame:
        """Retrieve all past subscan sessions.

        Parameters:
        ----------
        past_scan : pd.DataFrame or None, optional
            DataFrame containing past scan sessions. If None, it retrieves all past scans.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing all past subscan sessions.
        """
        if past_scan is None:
            past_scan = self.load_tunnel_logs()

        return past_scan.loc[past_scan["level"] == "TRACE"]

    def last_scan(self, past_scan: pd.DataFrame | None = None) -> pd.Series:
        """Retrieve the last scan session.

        Parameters:
        ----------
        past_scan : pd.DataFrame or None, optional
            DataFrame containing past scan sessions. If None, it retrieves all past scans.

        Returns:
        -------
        pd.Series
            The last scan session as a pandas Series.
        """
        if past_scan is None:
            past_scan = self.all_past_scans()

        return past_scan.iloc[-1, :]

    def last_sscan(self, past_sscan: pd.DataFrame | None = None) -> pd.Series:
        """Retrieve the last subscan session.

        Parameters:
        ----------
        past_sscan : pd.DataFrame or None, optional
            DataFrame containing past subscan sessions. If None, it retrieves all past subscans.

        Returns:
        -------
        pd.Series
            The last subscan session as a pandas Series.
        """
        if past_sscan is None:
            past_sscan = self.all_past_sscans()
        return past_sscan.iloc[-1, :]

    def get_last_state(self, tunnel_status: str | None = None, *, subscan: bool = False) -> pd.Series | None:
        """Retrieve the last state of the session or subscan.

        Parameters:
        ----------
        tunnel_status : str or None, optional
            The status of the tunnel. If "0", None is returned. Default is None.
        subscan : bool, optional
            Whether to retrieve the last state of a subscan. Default is False.

        Returns:
        -------
        pd.Series or None
            The last state of the session or subscan, or None if tunnel_status is "0".
        """
        last_state = self.last_sscan() if subscan else self.last_scan()
        if tunnel_status is None:
            tunnel_status = self.tunnel_status
        if tunnel_status =="0":
            return None
        return last_state
