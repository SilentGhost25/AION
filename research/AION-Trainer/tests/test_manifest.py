import time
import pytest
from pathlib import Path

from server.manifest import ManifestManager
from conftest import write_fake_file


def test_find_subject_path(manifest_manager, academic_root):
    path = manifest_manager.find_subject_path("BAI401")
    assert path is not None
    assert path.name == "BAI401"


def test_find_subject_path_returns_none_for_unknown_subject(manifest_manager):
    assert manifest_manager.find_subject_path("NOPE999") is None


def test_scan_with_no_files_returns_empty_manifest(manifest_manager):
    manifest = manifest_manager.scan("BAI401")
    assert manifest.books == []
    assert manifest.notes == []
    assert manifest.subject == "BAI401"


def test_scan_picks_up_files_in_correct_categories(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    write_fake_file(subject_dir / "textbooks" / "book1.pdf", "book content")
    write_fake_file(subject_dir / "notes" / "note1.docx", "note content")
    write_fake_file(subject_dir / "previous_papers" / "2023.pdf", "paper content")

    manifest = manifest_manager.scan("BAI401")

    assert len(manifest.books) == 1
    assert manifest.books[0].path.endswith("book1.pdf")
    assert len(manifest.notes) == 1
    assert len(manifest.previous_papers) == 1


def test_scan_ignores_unsupported_extensions(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    write_fake_file(subject_dir / "textbooks" / "readme.txt", "not supported")

    manifest = manifest_manager.scan("BAI401")
    assert manifest.books == []


def test_diff_detects_added_file(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    old_manifest = manifest_manager.scan("BAI401")

    write_fake_file(subject_dir / "notes" / "new_note.docx", "brand new")
    new_manifest = manifest_manager.scan("BAI401")

    diff = manifest_manager.diff(old_manifest, new_manifest)
    assert len(diff.added) == 1
    assert diff.added[0].path.endswith("new_note.docx")
    assert diff.modified == []
    assert diff.removed == []


def test_diff_detects_modified_file_via_hash_change(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    f = write_fake_file(subject_dir / "notes" / "note.docx", "version one")
    old_manifest = manifest_manager.scan("BAI401")

    f.write_text("version two — completely different content")
    new_manifest = manifest_manager.scan("BAI401")

    diff = manifest_manager.diff(old_manifest, new_manifest)
    assert len(diff.modified) == 1
    assert diff.added == []


def test_diff_detects_removed_file(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    f = write_fake_file(subject_dir / "notes" / "note.docx", "content")
    old_manifest = manifest_manager.scan("BAI401")

    f.unlink()
    new_manifest = manifest_manager.scan("BAI401")

    diff = manifest_manager.diff(old_manifest, new_manifest)
    assert len(diff.removed) == 1
    assert diff.added == []


def test_diff_has_changes_and_total_changed(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    old_manifest = manifest_manager.scan("BAI401")

    write_fake_file(subject_dir / "notes" / "a.docx", "a")
    write_fake_file(subject_dir / "notes" / "b.docx", "b")
    new_manifest = manifest_manager.scan("BAI401")

    diff = manifest_manager.diff(old_manifest, new_manifest)
    assert diff.has_changes is True
    assert diff.total_changed == 2


def test_save_bumps_version_and_persists(manifest_manager, academic_root):
    manifest = manifest_manager.scan("BAI401")
    assert manifest.version == "0.0"

    version = manifest_manager.save(manifest)
    assert version == "0.1"

    reloaded = manifest_manager.load_manifest("BAI401")
    assert reloaded.version == "0.1"


def test_get_or_create_returns_old_new_and_diff(manifest_manager, academic_root):
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    write_fake_file(subject_dir / "notes" / "note.docx", "content")

    old, new, diff = manifest_manager.get_or_create("BAI401")
    assert old is None
    assert len(new.notes) == 1
    assert len(diff.added) == 1
