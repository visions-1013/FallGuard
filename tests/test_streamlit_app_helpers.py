from app.streamlit_app import build_event, status_notice


def test_status_notice_keeps_english_state_and_chinese_message():
    notice = status_notice("standing")

    assert notice["state"] == "standing"
    assert notice["title"] == "当前状态"
    assert "站立" in notice["message"]


def test_status_notice_marks_fall_as_alert():
    notice = status_notice("fall")

    assert notice["state"] == "fall"
    assert notice["level"] == "error"
    assert "检测到摔倒" in notice["message"]


def test_build_event_uses_chinese_columns_and_english_state():
    event = build_event(12, "lying", "angle=80.0")

    assert event == {"帧号": "12", "状态": "lying", "说明": "angle=80.0"}
