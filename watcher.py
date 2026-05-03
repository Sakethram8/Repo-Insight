# new file: watcher.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import threading
from pathlib import Path
import falkordb

class GraphWatcher(FileSystemEventHandler):
    def __init__(self, repo_root: Path, graph: falkordb.Graph):
        self.repo_root = repo_root
        self.graph = graph
        self._debounce_timers: dict[str, threading.Timer] = {}
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            self._schedule_reingest(event.src_path)
    
    on_created = on_modified
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            rel = str(Path(event.src_path).relative_to(self.repo_root))
            self.graph.query(
                "UNWIND $fps AS fp MATCH (n {file_path: fp}) DETACH DELETE n",
                {"fps": [rel]}
            )
    
    def _schedule_reingest(self, abs_path: str):
        """Debounce: wait 800ms after last write before re-ingesting."""
        if abs_path in self._debounce_timers:
            self._debounce_timers[abs_path].cancel()
        
        def do_reingest():
            try:
                rel = str(Path(abs_path).relative_to(self.repo_root))
                from ingest import reingest_files
                reingest_files([rel], self.graph, self.repo_root)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Live re-ingest failed for %s: %s", abs_path, e
                )
            finally:
                self._debounce_timers.pop(abs_path, None)
        
        timer = threading.Timer(0.8, do_reingest)
        timer.daemon = True
        timer.start()
        self._debounce_timers[abs_path] = timer


def start_watcher(repo_root: Path, graph: falkordb.Graph) -> Observer:
    handler = GraphWatcher(repo_root, graph)
    observer = Observer()
    observer.schedule(handler, str(repo_root), recursive=True)
    observer.start()
    return observer  # caller must call observer.stop() on cleanup