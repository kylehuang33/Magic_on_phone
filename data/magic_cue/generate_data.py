import os, json, csv, zipfile, random, math, time
from datetime import datetime, timedelta, timezone

base = "./data/magic_cue"
os.makedirs(base, exist_ok=True)

# Helper to write JSONL
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Fixed dates around Aug 27, 2025 (Europe/London in summer is UTC+1).
LONDON_OFFSET = timedelta(hours=1)
def ld(dt_str):  # local London naive -> ISO 8601 with +01:00
    # dt_str like "2025-08-27 14:05"
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    dt = dt.replace(tzinfo=timezone(LONDON_OFFSET))
    return dt.isoformat()

# 1) notifications.jsonl (incoming message/call notifications)
notifications = [
    {
        "id": "notif_1001",
        "ts": ld("2025-08-27 14:05"),
        "app_pkg": "com.whatsapp",
        "channel": "messages",
        "title": "Alice",
        "text": "Are you free Tue evening?",
        "from_contact": {"name": "Alice", "importance": "high"},
        "conversation_id": "chat_alice",
        "screen_on": True
    },
    {
        "id": "notif_1002",
        "ts": ld("2025-08-27 14:12"),
        "app_pkg": "com.google.android.gm",
        "channel": "email",
        "title": "British Airways",
        "text": "BA082 arrives at 19:40 — LHR T5. Check-in opens 3 hours before departure.",
        "from_contact": {"name": "British Airways", "importance": "system"},
        "conversation_id": "mail_ba082",
        "screen_on": True
    },
    {
        "id": "notif_1003",
        "ts": ld("2025-08-27 15:21"),
        "app_pkg": "com.google.android.apps.messaging",
        "channel": "sms",
        "title": "George",
        "text": "When do you land?",
        "from_contact": {"name": "George", "importance": "high"},
        "conversation_id": "sms_george",
        "screen_on": True
    },
    {
        "id": "notif_1004",
        "ts": ld("2025-08-27 16:10"),
        "app_pkg": "com.whatsapp",
        "channel": "messages",
        "title": "Mum",
        "text": "Send the Brighton pics",
        "from_contact": {"name": "Mum", "importance": "high"},
        "conversation_id": "chat_mum",
        "screen_on": True
    },
    {
        "id": "notif_1005",
        "ts": ld("2025-08-27 11:02"),
        "app_pkg": "com.whatsapp",
        "channel": "messages",
        "title": "Team",
        "text": "Lunch now?",
        "from_contact": {"name": "Team Chat", "importance": "normal"},
        "conversation_id": "chat_team",
        "screen_on": True
    },
    {
        "id": "notif_1006",
        "ts": ld("2025-08-27 09:30"),
        "app_pkg": "com.amazon.mShop.android.shopping",
        "channel": "order",
        "title": "Amazon",
        "text": "Order 123-4567890-0001 delivered. Start a return?",
        "from_contact": {"name": "Amazon", "importance": "system"},
        "conversation_id": "ord_123456",
        "screen_on": False
    },
    {
        "id": "notif_1007",
        "ts": ld("2025-08-27 12:45"),
        "app_pkg": "com.whatsapp",
        "channel": "messages",
        "title": "Jamie",
        "text": "Dinner is at 221B Baker Street, NW1 at 7pm",
        "from_contact": {"name": "Jamie", "importance": "normal"},
        "conversation_id": "chat_jamie",
        "screen_on": True
    },
    {
        "id": "notif_1008",
        "ts": ld("2025-08-27 17:55"),
        "app_pkg": "com.google.android.apps.messaging",
        "channel": "sms",
        "title": "Spam",
        "text": "You won a prize! Click now.",
        "from_contact": {"name": "Unknown", "importance": "low"},
        "conversation_id": "sms_spam",
        "screen_on": True
    }
]
write_jsonl(os.path.join(base, "notifications.jsonl"), notifications)

# 2) calendar_events.json (user calendar, including flight)
calendar_events = [
    {
        "event_id": "cal_2001",
        "title": "Flight BA082 HKG → LHR",
        "start": ld("2025-08-27 13:00"),
        "end": ld("2025-08-27 19:40"),
        "location": "London Heathrow T5",
        "all_day": False
    },
    {
        "event_id": "cal_2002",
        "title": "Team Sync",
        "start": ld("2025-08-28 10:00"),
        "end": ld("2025-08-28 11:00"),
        "location": "Google Meet",
        "all_day": False
    },
    {
        "event_id": "cal_2003",
        "title": "Dinner with Jamie",
        "start": ld("2025-08-27 19:00"),
        "end": ld("2025-08-27 21:00"),
        "location": "221B Baker Street, NW1",
        "all_day": False
    },
    {
        "event_id": "cal_2004",
        "title": "Gym",
        "start": ld("2025-08-28 18:00"),
        "end": ld("2025-08-28 19:00"),
        "location": "Local Gym",
        "all_day": False
    },
    {
        "event_id": "cal_2005",
        "title": "Dentist",
        "start": ld("2025-09-02 09:30"),
        "end": ld("2025-09-02 10:00"),
        "location": "Smile Clinic",
        "all_day": False
    }
]
with open(os.path.join(base, "calendar_events.json"), "w", encoding="utf-8") as f:
    json.dump(calendar_events, f, ensure_ascii=False, indent=2)

# 3) photos_index.jsonl (albums / screenshots)
photos = [
    {
        "media_id": "photo_3001",
        "captured": ld("2025-07-12 16:40"),
        "album": "Brighton July",
        "tags": ["family", "beach", "sunset"],
        "uri": "content://media/external/images/media/3001"
    },
    {
        "media_id": "photo_3002",
        "captured": ld("2025-07-12 17:05"),
        "album": "Brighton July",
        "tags": ["pier", "friends"],
        "uri": "content://media/external/images/media/3002"
    },
    {
        "media_id": "photo_3003",
        "captured": ld("2025-08-26 09:23"),
        "album": "Screenshots",
        "tags": ["boarding-pass", "BA082"],
        "uri": "content://media/external/images/media/3003"
    },
    {
        "media_id": "photo_3004",
        "captured": ld("2025-08-27 11:55"),
        "album": "Receipts",
        "tags": ["amazon-return-label"],
        "uri": "content://media/external/images/media/3004"
    }
]
write_jsonl(os.path.join(base, "photos_index.jsonl"), photos)

# 4) facts_appsearch.jsonl (normalized “facts” extracted from the sources)
facts = [
    {
        "ns": "calendar",
        "id": "fact_flight_ba082",
        "kind": "flight",
        "title": "BA082 Arrival",
        "datetimeEpoch": int(datetime(2025,8,27,19,40,tzinfo=timezone(LONDON_OFFSET)).timestamp() * 1000),
        "entities": {"flightCode": "BA082", "airport": "LHR T5"},
        "sourceApp": "Calendar",
        "uri": "content://com.android.calendar/events/cal_2001"
    },
    {
        "ns": "messages",
        "id": "fact_address_dinner",
        "kind": "address",
        "title": "Dinner location",
        "datetimeEpoch": int(datetime(2025,8,27,19,0,tzinfo=timezone(LONDON_OFFSET)).timestamp() * 1000),
        "entities": {"address": "221B Baker Street, NW1, London"},
        "sourceApp": "WhatsApp",
        "uri": "geo:0,0?q=221B+Baker+Street+NW1+London"
    },
    {
        "ns": "photos",
        "id": "fact_album_brighton",
        "kind": "album",
        "title": "Brighton July album",
        "datetimeEpoch": int(datetime(2025,7,12,17,5,tzinfo=timezone(LONDON_OFFSET)).timestamp() * 1000),
        "entities": {"album": "Brighton July", "count": 2},
        "sourceApp": "MediaStore",
        "uri": "content://media/external/images/media?album=Brighton%20July"
    },
    {
        "ns": "orders",
        "id": "fact_order_123456",
        "kind": "order",
        "title": "Amazon Order 123-4567890-0001",
        "datetimeEpoch": int(datetime(2025,8,27,9,30,tzinfo=timezone(LONDON_OFFSET)).timestamp() * 1000),
        "entities": {"orderId": "123-4567890-0001", "status": "delivered"},
        "sourceApp": "Amazon",
        "uri": "app://amazon/order/123-4567890-0001"
    },
    {
        "ns": "calendar",
        "id": "fact_availability_Thu",
        "kind": "availability",
        "title": "Aug 28 evening mostly free",
        "datetimeEpoch": int(datetime(2025,8,28,18,0,tzinfo=timezone(LONDON_OFFSET)).timestamp() * 1000),
        "entities": {"freeBlocks": [["2025-08-28T17:00+01:00","2025-08-28T18:00+01:00"],
                                    ["2025-08-28T19:00+01:00","2025-08-28T21:00+01:00"]]},
        "sourceApp": "Calendar",
        "uri": "content://com.android.calendar/time/1756381200000?view=day"
    }
]
write_jsonl(os.path.join(base, "facts_appsearch.jsonl"), facts)

# 5) retrieval_pairs.jsonl (trigger -> retrieved facts with scores)
retrieval_pairs = [
    {
        "trigger_notif": "notif_1003",  # "When do you land?"
        "intent": "flight",
        "retrieved": [
            {"fact_id": "fact_flight_ba082", "cos_sim": 0.86, "rank": 1},
            {"fact_id": "fact_album_brighton", "cos_sim": 0.22, "rank": 2}
        ]
    },
    {
        "trigger_notif": "notif_1004",  # "Send the Brighton pics"
        "intent": "photos",
        "retrieved": [
            {"fact_id": "fact_album_brighton", "cos_sim": 0.91, "rank": 1}
        ]
    },
    {
        "trigger_notif": "notif_1001",  # "Are you free Tue evening?"
        "intent": "availability",
        "retrieved": [
            {"fact_id": "fact_availability_Thu", "cos_sim": 0.74, "rank": 1}
        ]
    },
    {
        "trigger_notif": "notif_1007",  # dinner address
        "intent": "address",
        "retrieved": [
            {"fact_id": "fact_address_dinner", "cos_sim": 0.88, "rank": 1}
        ]
    },
    {
        "trigger_notif": "notif_1008",
        "intent": "other",
        "retrieved": []
    }
]
write_jsonl(os.path.join(base, "retrieval_pairs.jsonl"), retrieval_pairs)

# 6) gate_training.csv (features -> label)
gate_rows = [
    # header
    ["intent","top_sim","avg_sim","n_facts","mins_since_last_same_intent","is_dnd","foreground_category","sender_priority","novelty","screen_on","label_clicked"],
    # flight: strong retrieval, important sender
    ["flight",0.86,0.54,2,180,0,"communication",1,0.82,1,1],
    # photos: very strong retrieval, high priority mom
    ["photos",0.91,0.91,1,1440,0,"communication",1,0.90,1,1],
    # availability: moderate similarity, screen on, accepted
    ["availability",0.74,0.74,1,720,0,"communication",1,0.70,1,1],
    # address: strong retrieval, accepted
    ["address",0.88,0.88,1,60,0,"communication",1,0.85,1,1],
    # team lunch: weak signal, user ignored
    ["availability",0.31,0.31,1,30,0,"productivity",0,0.20,1,0],
    # spam: no retrieval, dismissed
    ["other",0.00,0.00,0,9999,0,"other",0,0.00,1,0],
    # night DND: do not show
    ["photos",0.80,0.80,1,30,1,"communication",1,0.60,0,0],
    # repeated cue within cooldown: dismissed
    ["flight",0.84,0.52,2,5,0,"communication",1,0.10,1,0]
]

with open(os.path.join(base, "gate_training.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(gate_rows)

# 7) cues.jsonl (post-gate cue proposals with wording + actions)
cues = [
    {
        "cue_id": "cue_4001",
        "trigger_notif": "notif_1003",
        "gate_decision": True,
        "predicted_click_prob": 0.81,
        "title": "Flight lands 19:40 ✈️",
        "actions": [
            {"label": "Copy ETA", "deeplink": "app://cuecore/copy?text=Arrive%2019%3A40"},
            {"label": "Open itinerary", "deeplink": "content://com.android.calendar/events/cal_2001"}
        ]
    },
    {
        "cue_id": "cue_4002",
        "trigger_notif": "notif_1004",
        "gate_decision": True,
        "predicted_click_prob": 0.90,
        "title": "Share: Brighton July (2)",
        "actions": [
            {"label": "Open album", "deeplink": "content://media/external/images/media?album=Brighton%20July"},
            {"label": "Select & send", "deeplink": "app://photos/share?album=Brighton%20July&target=chat_mum"}
        ]
    },
    {
        "cue_id": "cue_4003",
        "trigger_notif": "notif_1001",
        "gate_decision": True,
        "predicted_click_prob": 0.67,
        "title": "Propose times: Thu 17:00–18:00 or 19:00–21:00",
        "actions": [
            {"label": "Open Calendar", "deeplink": "content://com.android.calendar/time/1756381200000?view=day"},
            {"label": "Copy times", "deeplink": "app://cuecore/copy?text=Thu%2017%3A00%E2%80%9318%3A00%20or%2019%3A00%E2%80%9321%3A00"}
        ]
    },
    {
        "cue_id": "cue_4004",
        "trigger_notif": "notif_1007",
        "gate_decision": True,
        "predicted_click_prob": 0.76,
        "title": "Open in Maps: 221B Baker Street",
        "actions": [
            {"label": "Open Maps", "deeplink": "geo:0,0?q=221B+Baker+Street+NW1+London"},
            {"label": "Copy address", "deeplink": "app://cuecore/copy?text=221B%20Baker%20Street%2C%20NW1%2C%20London"}
        ]
    }
]
write_jsonl(os.path.join(base, "cues.jsonl"), cues)

# 8) deep_links.json (templates)
deeplinks = {
    "calendar_day": "content://com.android.calendar/time/{epoch_ms}?view=day",
    "maps_query": "geo:0,0?q={query_urlenc}",
    "copy_text": "app://cuecore/copy?text={text_urlenc}",
    "photos_album": "content://media/external/images/media?album={album_urlenc}"
}
with open(os.path.join(base, "deep_links.json"), "w", encoding="utf-8") as f:
    json.dump(deeplinks, f, ensure_ascii=False, indent=2)

# # Write a small README
# readme = """# CueCore Synthetic Dataset

# This folder contains synthetic data to simulate a context-aware, proactive assistant:
# - notifications.jsonl — incoming messages/emails/orders
# - calendar_events.json — calendar items (including a flight)
# - photos_index.jsonl — albums/screenshots
# - facts_appsearch.jsonl — normalized facts (RAG index)
# - retrieval_pairs.jsonl — trigger → retrieved facts with similarity scores
# - gate_training.csv — features for a tiny cue/no-cue gate with labels
# - cues.jsonl — cue proposals (title + actions) after gating/LLM wording
# - deep_links.json — deeplink templates used by actions

# Timestamps are Europe/London (UTC+1 in August 2025).
# All content is purely synthetic.
# """
# with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
#     f.write(readme)

# # Zip everything for easy download
# zip_path = "/mnt/data/cuecore_synthetic_dataset.zip"
# with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
#     for fname in os.listdir(base):
#         z.write(os.path.join(base, fname), arcname=f"cuecore_synth/{fname}")