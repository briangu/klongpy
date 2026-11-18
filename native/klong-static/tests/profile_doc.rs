//! The profile document generator (tests/static/test_profile_doc.py, in Rust).
//! Reference: the checked-in page `../../docs/static-profile.md` (relative to this crate).
use klong_static::profile_doc::{check_text, generated_block, main, render_page, BEGIN, DEFAULT_PATH, END};
use klong_static::tables;

const PAGE: &str = "../../docs/static-profile.md";

fn run(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut o, mut e) = (Vec::new(), Vec::new());
    let rc = main(&a, &mut o, &mut e);
    (rc, String::from_utf8(o).unwrap(), String::from_utf8(e).unwrap())
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("klong-profile-doc-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn the_page_holds_the_generated_block_between_the_markers() {
    let page = render_page();
    assert_eq!(page.matches(BEGIN).count(), 1);
    assert_eq!(page.matches(END).count(), 1);
    assert!(page.contains(&generated_block()));
    assert!(check_text(&page));
    assert!(page.starts_with("# The static profile\n"));
}

#[test]
fn the_block_lists_every_row_form_and_rule() {
    let block = generated_block();
    for op in tables::OPTABLE {
        if op.allowed {
            assert!(block.contains(&format!("`{}`", op.id)), "{}", op.id);
        } else {
            assert!(block.contains(&format!("| {} | `{}` |", op.name, op.rule.unwrap())), "{}", op.id);
        }
    }
    for f in tables::FORMS {
        assert!(block.contains(&format!("`{}`", f.id)));
    }
    for r in tables::RULES {
        assert!(block.contains(&format!("| `{}` |", r.id)), "{}", r.id);
    }
}

#[test]
fn tables_are_well_formed_markdown() {
    let block = generated_block();
    for line in block.lines().filter(|l| l.starts_with('|')) {
        assert!(line.ends_with('|'), "{line}");
    }
    let tabs: Vec<&str> = block.split("\n\n").filter(|t| t.starts_with('|')).collect();
    assert!(!tabs.is_empty());
    for t in tabs {
        let widths: std::collections::HashSet<usize> =
            t.trim().lines().map(|r| r.replace("\\|", "").matches('|').count()).collect();
        assert_eq!(widths.len(), 1, "{}", t.lines().next().unwrap());
    }
}

#[test]
fn the_page_contains_the_hand_written_sections() {
    let page = render_page();
    for h in [
        "## Compile time and run time",
        "## Types",
        "## What a run-time function may contain",
        "## Typing rules",
        "## Host functions and entry points",
        "## Differences from KlongPy",
        "## Reference programs",
        "## Extra rules for FPGA kernels",
    ] {
        assert!(page.contains(h), "{h}");
    }
    assert!(page.contains("embedded.md#differences-from-the-interpreter"));
    assert!(page.contains("`(-7):%2` is -3"));
}

#[test]
fn check_is_0_when_fresh_1_after_a_hand_edit_2_when_missing() {
    let d = tmp("check");
    let path = d.join("static-profile.md");
    let p = path.to_str().unwrap();
    assert_eq!(run(&["--check", p]).0, 2);
    assert_eq!(run(&["--write", p]).0, 0);
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text, render_page());
    assert_eq!(run(&["--check", p]).0, 0);
    std::fs::write(&path, text.replace("| Plus |", "| Add |")).unwrap(); // inside the generated block
    let (rc, _, err) = run(&["--check", p]);
    assert_eq!(rc, 1);
    assert!(err.contains("out of date"));
    std::fs::write(&path, text.replace("## Compile time and run time", "## Stages")).unwrap(); // outside it: free
    assert_eq!(run(&["--check", p]).0, 0);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn write_refreshes_the_block_and_keeps_the_text_around_it() {
    let d = tmp("write");
    let path = d.join("p.md");
    let p = path.to_str().unwrap();
    std::fs::write(&path, format!("# Mine\n\n{BEGIN}\nstale\n{END}\n\nmy notes\n")).unwrap();
    assert_eq!(run(&["--write", p]).0, 0);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("# Mine\n"));
    assert!(text.ends_with("\nmy notes\n"));
    assert!(!text.contains("stale"));
    assert_eq!(run(&["--check", p]).0, 0);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn printing_the_page_and_bad_usage() {
    let (rc, out, _) = run(&[]);
    assert_eq!((rc, out), (0, render_page()));
    assert_eq!(run(&["--bogus"]).0, 2);
    assert_eq!(run(&["--check", "a", "b"]).0, 2);
    assert_eq!(DEFAULT_PATH, "docs/static-profile.md");
}

/// The checked-in page `../../docs/static-profile.md` has the generated block up to date, byte for byte.
/// (Its hand-written text has moved on from `HEAD` and `TAIL`: two paragraphs were added to the docs, so only the
/// block is compared, as `--check` does.)
#[test]
fn the_checked_in_page_is_up_to_date() {
    let text = std::fs::read_to_string(PAGE).expect("docs/static-profile.md, relative to the crate");
    assert!(check_text(&text), "run: klong profile-doc --write docs/static-profile.md");
    let a = text.find(BEGIN).unwrap() + BEGIN.len();
    let b = text.find(END).unwrap();
    assert_eq!(text[a..b].trim_matches('\n'), generated_block().trim_matches('\n'));
}
