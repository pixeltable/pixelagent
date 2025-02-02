from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type

import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter
from pixeltable.iterators import DocumentSplitter
from pixeltable.type_system import (
    Array,
    Audio,
    Bool,
    ColumnType,
    Document,
    Float,
    Image,
    Int,
    Json,
    String,
    Timestamp,
    Video,
    ArrayType,
    StringType,
)
from pydantic import BaseModel, Field


class PixelIndex(BaseModel):
    namespace: str = Field(
        default="audio_index", description="Namespace for the index tables"
    )
    table_name: str = Field(default="audio", description="Base name for the tables")
    index_type: str = Field(
        default="audio", description="Type of index ('audio' or 'pdf')"
    )
    clear_cache: bool = Field(
        default=False, description="Whether to recreate the index from scratch"
    )

    # Private fields - remove leading underscores for Pydantic
    main_table: Optional[Any] = Field(None, exclude=True)  # was audio_table
    chunks_view: Optional[Any] = Field(None, exclude=True)

    # Class-level constants with proper type annotations
    MEDIA_TYPES: ClassVar[Dict[Type, List[str]]] = {
        Audio: [".mp3", ".wav", ".ogg", ".m4a"],
        Image: [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
        Video: [".mp4", ".avi", ".mov", ".wmv"],
        Document: [".pdf", ".txt", ".doc", ".docx"],
    }

    PYTHON_TO_PIXEL_TYPES: ClassVar[Dict[Type, ColumnType]] = {
        bool: Bool,
        int: Int,
        float: Float,
        str: String,
        dict: Json,
        datetime: Timestamp,
        bytes: String,
    }

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize the index after validation"""
        self._setup_tables()

    @property
    def table_path(self) -> str:
        return f"{self.namespace}.{self.table_name}"

    @property
    def chunks_path(self) -> str:
        return f"{self.namespace}.{self.table_name}_sentence_chunks"

    @property
    def file_column(self) -> str:
        """Dynamically determine the file column name based on the index type."""
        return "audio_file" if self.index_type == "audio" else "pdf"

    def _setup_tables(self) -> None:
        """Set up the main table and chunks view"""
        if self.clear_cache:
            self._cleanup_existing_tables()

        self._setup_main_table()  # Renamed method
        self._setup_chunks_view()

    def _cleanup_existing_tables(self) -> None:
        """Remove existing tables if clear_cache is True"""
        if self.table_path in pxt.list_tables():
            pxt.drop_table(self.table_path, force=True, if_not_exists="ignore")

    def _setup_main_table(self) -> None:
        """Create or get the main table based on index_type"""
        if self.table_path not in pxt.list_tables():
            pxt.create_dir(self.namespace, if_exists="ignore")
            if self.index_type == "audio":
                self.main_table = pxt.create_table(self.table_path, {self.file_column: Audio})
                self.main_table.add_computed_column(
                    transcription=whisper.transcribe(
                        audio=getattr(self.main_table, self.file_column),
                        model="base.en"
                    )
                )
            elif self.index_type == "pdf":
                self.main_table = pxt.create_table(self.table_path, {self.file_column: Document})
            else:
                raise ValueError(f"Unsupported index_type: {self.index_type}")
        else:
            self.main_table = pxt.get_table(self.table_path)

    def _setup_chunks_view(self) -> None:
        """Create or get the sentence/document chunks view"""
        if self.chunks_path not in pxt.list_tables():
            if self.index_type == "audio":
                self.chunks_view = pxt.create_view(
                    self.chunks_path,
                    self.main_table,
                    iterator=StringSplitter.create(
                        text=self.main_table.transcription.text, separators="sentence"
                    ),
                )
            elif self.index_type == "pdf":
                self.chunks_view = pxt.create_view(
                    self.chunks_path,
                    self.main_table,
                    iterator=DocumentSplitter.create(
                        document=getattr(self.main_table, self.file_column),
                        separators="token_limit",
                        limit=300
                    ),
                )
            else:
                raise ValueError(f"Unsupported index_type: {self.index_type}")
            self.chunks_view.add_embedding_index(
                column="text",
                string_embed=sentence_transformer.using(model_id="intfloat/e5-large-v2")
            )
        else:
            self.chunks_view = pxt.get_table(self.chunks_path)

    def _infer_column_type(self, value: Any) -> ColumnType:
        """Infer PixelTable column type from a Python value"""
        # Handle lists
        if isinstance(value, list):
            if not value:
                # For empty lists, default to string array
                return ArrayType(dtype=StringType(), nullable=False)
            if isinstance(value[0], str):
                # For string lists, use Array of String
                return ArrayType(dtype=StringType(), nullable=False)
            # For other types, recursively infer the element type
            element_type = self._infer_column_type(value[0])
            return ArrayType(dtype=element_type, nullable=False)

        # Handle numpy float types
        if str(type(value).__module__).startswith("numpy"):
            return Float

        # Handle file extensions for media types
        if isinstance(value, str):
            lower_val = value.lower()
            for type_cls, extensions in self.MEDIA_TYPES.items():
                if any(lower_val.endswith(ext) for ext in extensions):
                    return type_cls

        # Get type from mapping or default to String
        return self.PYTHON_TO_PIXEL_TYPES.get(type(value), String)

    def insert(self, file_path: str, metadata: Optional[Dict] = None) -> None:
        """Insert a file (audio or pdf) with optional metadata"""
        if metadata:
            # Add new columns for metadata if needed
            new_columns = {
                col: self._infer_column_type(val)
                for col, val in metadata.items()
                if col not in self.main_table.get_metadata()["schema"]
            }
            if new_columns:
                self.main_table.add_columns(new_columns, if_exists="ignore")

        self.main_table.insert([{self.file_column: file_path, **(metadata or {})}])

    def search(
        self,
        semantic_query: Optional[str] = None,
        keyword: Optional[str] = None,
        metadata_filters: Optional[Dict] = None,
        min_similarity: float = 0.5,
        limit: int = 20,
    ) -> Any:
        """Search the index using semantic search, keywords, and metadata filters"""
        query = self.chunks_view
        select_cols = [self.chunks_view.text, getattr(self.chunks_view, self.file_column)]

        if semantic_query:
            sim = self.chunks_view.text.similarity(semantic_query)
            query = query.where(sim >= min_similarity).order_by(sim, asc=False)
            select_cols.append(("similarity", sim))

        if keyword:
            query = query.where(
                self.chunks_view.text.contains(f"{keyword}")
            )

        if metadata_filters:
            for field, value in metadata_filters.items():
                query = query.where(getattr(self.chunks_view, field) == value)
                select_cols.append(getattr(self.chunks_view, field))

        return query.select(*select_cols).limit(limit).collect()
